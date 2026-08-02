"""
Tests for remote_workflow_node.py.

External dependencies mocked:
  - folder_paths  (ComfyUI runtime, not installable standalone — mocked in conftest.py)
  - requests      (HTTP calls)
  - websocket     (WebSocket connection)
  - cv2           (video encoding in upload_video_to_remote)
"""
import json
import threading
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import numpy as np

from remote_workflow_node import RemoteWorkflowExecutor, _WORKFLOW_CACHE_MAX


# ─────────────────────────── factories ────────────────────────────────────────

def make_executor() -> RemoteWorkflowExecutor:
    return RemoteWorkflowExecutor()


def api_workflow(*node_ids: str, class_types: dict | None = None) -> dict:
    """Return a minimal API-format workflow dict (keys are digit strings)."""
    class_types = class_types or {}
    return {
        nid: {"class_type": class_types.get(nid, "KSampler"), "inputs": {}}
        for nid in node_ids
    }


def ws_message_sequence(prompt_id: str, outputs: dict) -> list[str]:
    """Build the ComfyUI WebSocket message sequence for a successful run."""
    msgs = [
        json.dumps({
            "type": "executed",
            "data": {"node": node_id, "prompt_id": prompt_id, "output": output},
        })
        for node_id, output in outputs.items()
    ]
    msgs.append(json.dumps({
        "type": "executing",
        "data": {"node": None, "prompt_id": prompt_id},
    }))
    return msgs


class FakeWebSocket:
    """Delivers messages synchronously in the ws_thread, then fires on_close."""

    def __init__(self, messages: list[str], on_message=None, on_error=None, on_close=None):
        self._messages = messages
        self.on_message = on_message
        self.on_close = on_close

    def run_forever(self):
        for msg in self._messages:
            if self.on_message:
                self.on_message(self, msg)
        if self.on_close:
            self.on_close(self, None, None)

    def close(self):
        pass


def fake_ws_factory(messages: list[str]):
    """Return a factory compatible with `websocket.WebSocketApp(url, **cbs)`."""
    def factory(url, on_message=None, on_error=None, on_close=None):
        return FakeWebSocket(messages, on_message=on_message, on_close=on_close)
    return factory


# ─────────────────────────── load_workflow ────────────────────────────────────

class TestLoadWorkflow:

    def test_valid_json_returns_dict(self):
        ex = make_executor()
        wf = api_workflow("1", "2")
        result = ex.load_workflow(json.dumps(wf))
        assert result == wf

    def test_invalid_json_returns_none(self):
        ex = make_executor()
        assert ex.load_workflow("not json at all") is None

    def test_empty_string_returns_none(self):
        ex = make_executor()
        assert ex.load_workflow("") is None

    def test_same_string_returns_cached_object(self):
        ex = make_executor()
        s = json.dumps(api_workflow("1"))
        first = ex.load_workflow(s)
        second = ex.load_workflow(s)
        assert first is second

    def test_cache_evicts_oldest_when_full(self):
        ex = make_executor()
        keys = []
        for i in range(_WORKFLOW_CACHE_MAX + 1):
            wf = {str(i): {"class_type": "X", "inputs": {}}}
            s = json.dumps(wf)
            ex.load_workflow(s)
            keys.append(s)

        assert len(ex.workflow_cache) == _WORKFLOW_CACHE_MAX
        assert keys[0] not in ex.workflow_cache   # oldest evicted
        assert keys[-1] in ex.workflow_cache       # newest present

    def test_returns_same_cached_reference(self):
        """load_workflow returns a direct cache reference; execute_remote deepcopies it."""
        ex = make_executor()
        s = json.dumps(api_workflow("1"))
        first = ex.load_workflow(s)
        second = ex.load_workflow(s)
        assert first is second  # same object — deepcopy protection is execute_remote's job


# ─────────────────────────── modify_workflow_input ────────────────────────────

class TestModifyWorkflowInput:

    # ── image ─────────────────────────────────────────────────────────────────

    def test_image_injects_into_load_image_node(self):
        ex = make_executor()
        wf = {"5": {"class_type": "LoadImage", "inputs": {}}}
        with patch.object(ex, "upload_image_to_remote", return_value="up.png"):
            result = ex.modify_workflow_input(wf, "5", "image", torch.zeros((1, 4, 4, 3)), "h:8188")
        assert result["5"]["inputs"]["image"] == "up.png"

    def test_image_skips_wrong_class(self):
        ex = make_executor()
        wf = {"5": {"class_type": "KSampler", "inputs": {}}}
        with patch.object(ex, "upload_image_to_remote", return_value="up.png") as m:
            ex.modify_workflow_input(wf, "5", "image", torch.zeros((1, 4, 4, 3)), "h:8188")
        m.assert_not_called()

    def test_image_upload_failure_leaves_node_unchanged(self):
        ex = make_executor()
        wf = {"5": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
        with patch.object(ex, "upload_image_to_remote", return_value=None):
            result = ex.modify_workflow_input(wf, "5", "image", torch.zeros((1, 4, 4, 3)), "h:8188")
        assert result["5"]["inputs"]["image"] == "old.png"

    # ── video ─────────────────────────────────────────────────────────────────

    def test_video_injects_into_load_video_node(self):
        ex = make_executor()
        wf = {"7": {"class_type": "LoadVideo", "inputs": {}}}
        with patch.object(ex, "upload_video_to_remote", return_value="clip.mp4"):
            result = ex.modify_workflow_input(wf, "7", "video", torch.zeros((8, 4, 4, 3)), "h:8188")
        assert result["7"]["inputs"]["video"] == "clip.mp4"

    def test_video_falls_back_to_load_image_with_first_frame(self):
        ex = make_executor()
        wf = {"7": {"class_type": "LoadImage", "inputs": {}}}
        with patch.object(ex, "upload_image_to_remote", return_value="frame.png"):
            result = ex.modify_workflow_input(wf, "7", "video", torch.zeros((8, 4, 4, 3)), "h:8188")
        assert result["7"]["inputs"]["image"] == "frame.png"

    def test_video_skips_unrelated_class(self):
        ex = make_executor()
        wf = {"7": {"class_type": "KSampler", "inputs": {}}}
        with patch.object(ex, "upload_video_to_remote", return_value="clip.mp4") as m:
            ex.modify_workflow_input(wf, "7", "video", torch.zeros((8, 4, 4, 3)), "h:8188")
        m.assert_not_called()

    # ── text ──────────────────────────────────────────────────────────────────

    def test_text_updates_existing_text_field(self):
        ex = make_executor()
        wf = {"3": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}}}
        ex.modify_workflow_input(wf, "3", "text", "new prompt", "h:8188")
        assert wf["3"]["inputs"]["text"] == "new prompt"

    def test_text_updates_existing_prompt_field(self):
        ex = make_executor()
        wf = {"3": {"class_type": "CLIPTextEncode", "inputs": {"prompt": "old"}}}
        ex.modify_workflow_input(wf, "3", "text", "hello", "h:8188")
        assert wf["3"]["inputs"]["prompt"] == "hello"

    def test_text_falls_back_to_prompt_when_no_known_field(self):
        ex = make_executor()
        wf = {"3": {"class_type": "CLIPTextEncode", "inputs": {}}}
        ex.modify_workflow_input(wf, "3", "text", "hello", "h:8188")
        assert wf["3"]["inputs"]["prompt"] == "hello"

    def test_text_priority_order(self):
        """'prompt' has lower priority than 'text' when both are present."""
        ex = make_executor()
        wf = {"3": {"class_type": "X", "inputs": {"prompt": "a", "text": "b"}}}
        ex.modify_workflow_input(wf, "3", "text", "new", "h:8188")
        assert wf["3"]["inputs"]["text"] == "new"
        assert wf["3"]["inputs"]["prompt"] == "a"  # untouched

    # ── audio ─────────────────────────────────────────────────────────────────

    def test_audio_injects_into_load_audio_node(self):
        ex = make_executor()
        wf = {"9": {"class_type": "LoadAudio", "inputs": {}}}
        audio = {"waveform": torch.zeros((1, 2, 44100)), "sample_rate": 44100}
        with patch.object(ex, "upload_audio_to_remote", return_value="track.wav"):
            result = ex.modify_workflow_input(wf, "9", "audio", audio, "h:8188")
        assert result["9"]["inputs"]["audio"] == "track.wav"

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_missing_node_id_returns_workflow_unchanged(self):
        ex = make_executor()
        wf = {"1": {"class_type": "LoadImage", "inputs": {}}}
        result = ex.modify_workflow_input(wf, "999", "image", torch.zeros((1, 4, 4, 3)), "h:8188")
        assert result == wf


# ─────────────────────────── upload_image_to_remote ───────────────────────────

class TestUploadImage:

    def _ok_response(self, name="out.png", subfolder=""):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"name": name, "subfolder": subfolder}
        return resp

    def test_returns_filename_without_subfolder(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response("out.png", "")):
            result = ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3)))
        assert result == "out.png"

    def test_returns_path_with_subfolder(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response("out.png", "sub")):
            result = ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3)))
        assert result == "sub/out.png"

    def test_http_error_returns_none(self):
        ex = make_executor()
        resp = MagicMock(); resp.status_code = 500
        with patch("requests.post", return_value=resp):
            assert ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3))) is None

    def test_unique_filename_each_call(self):
        ex = make_executor()
        names: list[str] = []

        def capture(url, files, data, timeout):
            names.append(files["image"][0])
            return self._ok_response()

        with patch("requests.post", side_effect=capture):
            ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3)))
            ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3)))

        assert names[0] != names[1]

    def test_4d_batch_tensor_accepted(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response()):
            result = ex.upload_image_to_remote("h:8188", torch.zeros((4, 8, 8, 3)))
        assert result is not None

    def test_3d_single_tensor_accepted(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response()):
            result = ex.upload_image_to_remote("h:8188", torch.zeros((8, 8, 3)))
        assert result is not None

    def test_wrong_shape_returns_none(self):
        ex = make_executor()
        assert ex.upload_image_to_remote("h:8188", torch.zeros((64,))) is None

    def test_network_exception_returns_none(self):
        ex = make_executor()
        with patch("requests.post", side_effect=ConnectionError("refused")):
            assert ex.upload_image_to_remote("h:8188", torch.zeros((1, 4, 4, 3))) is None


# ─────────────────────────── wait_for_completion ──────────────────────────────

class TestWaitForCompletion:

    def _run(self, pid: str, messages: list[str], timeout: int = 5):
        ex = make_executor()
        with patch("websocket.WebSocketApp", side_effect=fake_ws_factory(messages)):
            return ex.wait_for_completion("h:8188", pid, timeout=timeout)

    def test_successful_run_returns_output_dict(self):
        pid = "prompt-ok"
        outputs = {"2": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}
        result = self._run(pid, ws_message_sequence(pid, outputs))

        assert result is not None
        assert "2" in result
        assert result["2"]["images"][0]["filename"] == "a.png"

    def test_all_node_outputs_collected(self):
        pid = "prompt-multi"
        outputs = {
            "1": {"text": ["hello"]},
            "2": {"images": [{"filename": "b.png", "subfolder": "", "type": "output"}]},
        }
        result = self._run(pid, ws_message_sequence(pid, outputs))

        assert "1" in result
        assert "2" in result

    def test_execution_error_returns_none(self):
        pid = "prompt-err"
        messages = [json.dumps({
            "type": "execution_error",
            "data": {"node": "3", "prompt_id": pid, "error": "node crashed"},
        })]
        assert self._run(pid, messages) is None

    def test_ignores_messages_from_other_prompt_ids(self):
        my_pid = "mine"
        other_pid = "theirs"
        messages = (
            # noise from a concurrent prompt
            [json.dumps({"type": "executed",
                         "data": {"node": "99", "prompt_id": other_pid, "output": {"text": ["wrong"]}}})]
            + ws_message_sequence(my_pid, {"1": {"text": ["correct"]}})
        )
        result = self._run(my_pid, messages)

        assert result is not None
        assert "99" not in result
        assert "1" in result

    def test_timeout_returns_none(self):
        """If done_event is never set the call must return None, not hang."""
        ex = make_executor()
        # WebSocket that never fires on_close or the executing(None) message
        def eternal_ws(url, on_message=None, on_error=None, on_close=None):
            ws = MagicMock()
            ws.run_forever.side_effect = lambda: threading.Event().wait(60)
            ws.close.side_effect = lambda: None
            return ws

        with patch("websocket.WebSocketApp", side_effect=eternal_ws):
            result = ex.wait_for_completion("h:8188", "pid-x", timeout=0.1)

        assert result is None

    def test_binary_messages_are_ignored(self):
        """Binary frames (progress previews) must not crash the handler."""
        pid = "prompt-bin"
        binary_msg = b"\x00\x01\x02\x03"
        messages_with_binary = [binary_msg] + ws_message_sequence(pid, {"1": {"text": ["ok"]}})

        ex = make_executor()

        def ws_factory(url, on_message=None, on_error=None, on_close=None):
            return FakeWebSocket(messages_with_binary, on_message=on_message, on_close=on_close)

        with patch("websocket.WebSocketApp", side_effect=ws_factory):
            result = ex.wait_for_completion("h:8188", pid, timeout=5)

        assert result is not None


# ─────────────────────────── execute_remote (integration) ─────────────────────

class TestExecuteRemote:
    """Patches all I/O; exercises routing, error messages, and output assembly."""

    HOST = "h"
    PORT = 8188
    TIMEOUT = 60

    def _base_wf(self, class_types: dict | None = None) -> dict:
        return api_workflow("1", "2", class_types=class_types or {"2": "SaveImage"})

    def _call(self, ex, wf_json, selected, **kwargs):
        return ex.execute_remote(self.HOST, self.PORT, self.TIMEOUT, wf_json, selected, **kwargs)

    # ── early-exit error paths ─────────────────────────────────────────────────

    def test_connection_failure(self):
        ex = make_executor()
        with patch.object(ex, "test_remote_connection", return_value=False):
            _, text, _, _ = self._call(ex, "{}", "{}")
        assert "Cannot connect" in text

    def test_invalid_workflow_json(self):
        ex = make_executor()
        with patch.object(ex, "test_remote_connection", return_value=True):
            _, text, _, _ = self._call(ex, "NOT JSON", "{}")
        assert "Failed to load workflow" in text

    def test_non_api_format_workflow(self):
        ex = make_executor()
        wf = json.dumps({"nodes": [], "links": []})
        with patch.object(ex, "test_remote_connection", return_value=True):
            _, text, _, _ = self._call(ex, wf, "{}")
        assert "API format" in text

    def test_invalid_selected_nodes_json(self):
        ex = make_executor()
        wf = json.dumps(self._base_wf())
        with patch.object(ex, "test_remote_connection", return_value=True):
            _, text, _, _ = self._call(ex, wf, "BROKEN")
        assert "Invalid" in text

    def test_empty_selected_nodes(self):
        ex = make_executor()
        wf = json.dumps(self._base_wf())
        with patch.object(ex, "test_remote_connection", return_value=True):
            _, text, _, _ = self._call(ex, wf, "{}")
        assert "No nodes selected" in text

    def test_queue_prompt_failure(self):
        ex = make_executor()
        wf = json.dumps(self._base_wf())
        selected = json.dumps({"1": "text"})
        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value=None),
        ):
            _, text, _, _ = self._call(ex, wf, selected)
        assert "Failed to submit" in text

    def test_execution_timeout(self):
        ex = make_executor()
        wf = json.dumps(self._base_wf())
        selected = json.dumps({"1": "text"})
        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value=None),
        ):
            _, text, _, _ = self._call(ex, wf, selected)
        assert "timed out" in text.lower() or "failed" in text.lower()

    # ── output assembly ───────────────────────────────────────────────────────

    def test_successful_image_output(self):
        ex = make_executor()
        wf = self._base_wf({"2": "SaveImage"})
        wf_json = json.dumps(wf)
        selected = json.dumps({"1": "text"})

        fake_img = torch.ones((1, 8, 8, 3))
        fake_outputs = {
            "2": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}
        }

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value=fake_outputs),
            patch.object(ex, "download_output_file", return_value=(fake_img, None)),
        ):
            img, text, audio, video = self._call(ex, wf_json, selected)

        assert img.shape == (1, 8, 8, 3)
        assert text == "Execution successful"

    def test_multi_image_batch_concatenated(self):
        ex = make_executor()
        wf = self._base_wf({"2": "SaveImage"})
        wf_json = json.dumps(wf)
        selected = json.dumps({"1": "text"})

        img_a = torch.zeros((1, 8, 8, 3))
        img_b = torch.ones((1, 8, 8, 3))
        fake_outputs = {
            "2": {"images": [
                {"filename": "a.png", "subfolder": "", "type": "output"},
                {"filename": "b.png", "subfolder": "", "type": "output"},
            ]}
        }

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value=fake_outputs),
            patch.object(ex, "download_output_file", side_effect=[(img_a, None), (img_b, None)]),
        ):
            img, _, _, _ = self._call(ex, wf_json, selected)

        assert img.shape[0] == 2  # batch preserved

    def test_multi_text_outputs_joined(self):
        ex = make_executor()
        wf = self._base_wf({"2": "easy showAnything"})
        wf_json = json.dumps(wf)
        selected = json.dumps({"1": "text"})

        fake_outputs = {"2": {"text": ["line one", "line two"]}}

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value=fake_outputs),
        ):
            _, text, _, _ = self._call(ex, wf_json, selected)

        assert "line one" in text
        assert "line two" in text

    def test_no_output_nodes_returns_defaults(self):
        ex = make_executor()
        wf = self._base_wf()
        wf_json = json.dumps(wf)
        selected = json.dumps({"1": "text"})

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value={"1": {}}),
        ):
            img, text, _, video = self._call(ex, wf_json, selected)

        assert img.shape == (1, 64, 64, 3)
        assert text == "Execution successful"
        assert video.shape == (1, 64, 64, 3)

    # ── cache immutability ────────────────────────────────────────────────────

    def test_cached_workflow_not_mutated_between_runs(self):
        """Running the node twice must not carry injected values across runs."""
        ex = make_executor()
        wf = {"1": {"class_type": "LoadImage", "inputs": {"image": "original.png"}}}
        wf_json = json.dumps(wf)
        selected = json.dumps({"1": "image"})
        tensor = torch.zeros((1, 4, 4, 3))

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "upload_image_to_remote", return_value="injected.png"),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value={}),
        ):
            self._call(ex, wf_json, selected, image_1=tensor)
            self._call(ex, wf_json, selected, image_1=tensor)

        cached = ex.workflow_cache[wf_json]
        assert cached["1"]["inputs"]["image"] == "original.png"

    # ── multi-input routing ───────────────────────────────────────────────────

    def test_multiple_image_inputs_mapped_to_correct_nodes(self):
        """image_1 → node 3, image_2 → node 7 (sorted by node id)."""
        ex = make_executor()
        wf = {
            "3": {"class_type": "LoadImage", "inputs": {}},
            "7": {"class_type": "LoadImage", "inputs": {}},
            "9": {"class_type": "SaveImage", "inputs": {}},
        }
        wf_json = json.dumps(wf)
        selected = json.dumps({"3": "image", "7": "image"})

        calls: list[tuple] = []

        def capture_modify(workflow, node_id, input_type, value, server):
            calls.append((node_id, input_type))
            return workflow

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "modify_workflow_input", side_effect=capture_modify),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value={}),
        ):
            img1 = torch.zeros((1, 4, 4, 3))
            img2 = torch.ones((1, 4, 4, 3))
            self._call(ex, wf_json, selected, image_1=img1, image_2=img2)

        assert calls[0] == ("3", "image")
        assert calls[1] == ("7", "image")

    def test_mixed_type_inputs_routed_independently(self):
        """image_1 and text_1 come from separate counters per type."""
        ex = make_executor()
        wf = {
            "2": {"class_type": "LoadImage", "inputs": {}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {}},
            "8": {"class_type": "SaveImage", "inputs": {}},
        }
        wf_json = json.dumps(wf)
        selected = json.dumps({"2": "image", "5": "text"})

        calls: list[tuple] = []

        def capture_modify(workflow, node_id, input_type, value, server):
            calls.append((node_id, input_type))
            return workflow

        with (
            patch.object(ex, "test_remote_connection", return_value=True),
            patch.object(ex, "modify_workflow_input", side_effect=capture_modify),
            patch.object(ex, "queue_prompt", return_value="pid"),
            patch.object(ex, "wait_for_completion", return_value={}),
        ):
            self._call(
                ex, wf_json, selected,
                image_1=torch.zeros((1, 4, 4, 3)),
                text_1="hello",
            )

        assert ("2", "image") in calls
        assert ("5", "text") in calls


# ─────────────────────────── upload_mask_to_remote ────────────────────────────

class TestUploadMask:

    def _ok_response(self, name="mask.png", subfolder=""):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"name": name, "subfolder": subfolder}
        return resp

    def test_2d_tensor_accepted(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response("m.png")):
            result = ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8)))
        assert result == "m.png"

    def test_3d_tensor_squeezed_to_2d(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response("m.png")) as mock_post:
            result = ex.upload_mask_to_remote("h:8188", torch.zeros((1, 8, 8)))
        assert result is not None

    def test_wrong_shape_returns_none(self):
        ex = make_executor()
        assert ex.upload_mask_to_remote("h:8188", torch.zeros((64,))) is None

    def test_4d_tensor_returns_none(self):
        ex = make_executor()
        assert ex.upload_mask_to_remote("h:8188", torch.zeros((1, 1, 8, 8))) is None

    def test_returns_path_with_subfolder(self):
        ex = make_executor()
        with patch("requests.post", return_value=self._ok_response("m.png", "sub")):
            result = ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8)))
        assert result == "sub/m.png"

    def test_unique_filename_each_call(self):
        ex = make_executor()
        names: list[str] = []

        def capture(url, files, data, timeout):
            names.append(files["image"][0])
            return self._ok_response()

        with patch("requests.post", side_effect=capture):
            ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8)))
            ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8)))

        assert names[0] != names[1]

    def test_http_error_returns_none(self):
        ex = make_executor()
        resp = MagicMock(); resp.status_code = 500
        with patch("requests.post", return_value=resp):
            assert ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8))) is None

    def test_network_exception_returns_none(self):
        ex = make_executor()
        with patch("requests.post", side_effect=ConnectionError("refused")):
            assert ex.upload_mask_to_remote("h:8188", torch.zeros((8, 8))) is None

    def test_uploads_as_rgb_png(self):
        """Mask must be uploaded as RGB so any channel extraction works on the remote."""
        ex = make_executor()
        captured = {}

        def capture(url, files, data, timeout):
            raw_bytes = files["image"][1]
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(raw_bytes))
            captured["mode"] = img.mode
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"name": "m.png", "subfolder": ""}
            return resp

        with patch("requests.post", side_effect=capture):
            ex.upload_mask_to_remote("h:8188", torch.full((8, 8), 0.5))

        assert captured["mode"] == "RGB"


# ─────────────────────────── modify_workflow_input (mask) ─────────────────────

class TestModifyWorkflowInputMask:

    def test_mask_injects_into_load_image_mask_node(self):
        ex = make_executor()
        wf = {"4": {"class_type": "LoadImageMask", "inputs": {}}}
        with patch.object(ex, "upload_mask_to_remote", return_value="m.png"):
            result = ex.modify_workflow_input(wf, "4", "mask", torch.zeros((8, 8)), "h:8188")
        assert result["4"]["inputs"]["image"] == "m.png"

    def test_mask_sets_default_channel_red(self):
        ex = make_executor()
        wf = {"4": {"class_type": "LoadImageMask", "inputs": {}}}
        with patch.object(ex, "upload_mask_to_remote", return_value="m.png"):
            result = ex.modify_workflow_input(wf, "4", "mask", torch.zeros((8, 8)), "h:8188")
        assert result["4"]["inputs"]["channel"] == "red"

    def test_mask_preserves_existing_channel(self):
        ex = make_executor()
        wf = {"4": {"class_type": "LoadImageMask", "inputs": {"channel": "alpha"}}}
        with patch.object(ex, "upload_mask_to_remote", return_value="m.png"):
            result = ex.modify_workflow_input(wf, "4", "mask", torch.zeros((8, 8)), "h:8188")
        assert result["4"]["inputs"]["channel"] == "alpha"

    def test_mask_skips_non_mask_node(self):
        ex = make_executor()
        wf = {"4": {"class_type": "KSampler", "inputs": {}}}
        with patch.object(ex, "upload_mask_to_remote", return_value="m.png") as mock_up:
            ex.modify_workflow_input(wf, "4", "mask", torch.zeros((8, 8)), "h:8188")
        mock_up.assert_not_called()

    def test_mask_upload_failure_leaves_node_unchanged(self):
        ex = make_executor()
        wf = {"4": {"class_type": "LoadImageMask", "inputs": {"image": "old.png"}}}
        with patch.object(ex, "upload_mask_to_remote", return_value=None):
            result = ex.modify_workflow_input(wf, "4", "mask", torch.zeros((8, 8)), "h:8188")
        assert result["4"]["inputs"]["image"] == "old.png"
