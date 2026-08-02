import requests
import json
import copy
import io
import os
import time
import uuid
import threading
import numpy as np
from PIL import Image
import torch
import folder_paths

_WORKFLOW_CACHE_MAX = 10

class RemoteWorkflowExecutor:

    def __init__(self):
        self.client_id = str(uuid.uuid4())
        self.workflow_cache = {}   # json_str → parsed dict

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_host": ("STRING", {"default": "192.168.1.100"}),
                "remote_port": ("INT", {"default": 8188, "min": 1, "max": 65535}),
                "timeout": ("INT", {"default": 600, "min": 10, "max": 3600}),
                "workflow_file": ("STRING", {"default": "", "multiline": True}),
                "selected_nodes": ("STRING", {"default": "{}", "multiline": True}),
                "saved_state": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "mask_1": ("MASK",),
                "text_1": ("STRING", {"forceInput": True}),
                "audio_1": ("AUDIO",),
                "video_1": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "IMAGE")
    RETURN_NAMES = ("output_image", "output_text", "output_audio", "output_video")
    FUNCTION = "execute_remote"
    CATEGORY = "remote_nodes"
    
    def load_workflow(self, workflow_json_str):
        if workflow_json_str in self.workflow_cache:
            return self.workflow_cache[workflow_json_str]

        try:
            workflow = json.loads(workflow_json_str)
            if len(self.workflow_cache) >= _WORKFLOW_CACHE_MAX:
                oldest_key = next(iter(self.workflow_cache))
                del self.workflow_cache[oldest_key]
            self.workflow_cache[workflow_json_str] = workflow
            return workflow
        except Exception:
            return None

    def get_workflow_nodes(self, workflow):
        nodes_info = []
        if workflow and isinstance(workflow, dict):
            for node_id, node_data in workflow.items():
                if isinstance(node_data, dict) and "class_type" in node_data:
                    nodes_info.append({
                        "id": node_id,
                        "class_type": node_data.get("class_type", "Unknown"),
                        "inputs": node_data.get("inputs", {})
                    })
        return nodes_info
    
    def create_empty_audio(self):
        return {
            "waveform": torch.zeros((1, 2, 44100)),
            "sample_rate": 44100
        }
    
    def test_remote_connection(self, server_address):
        try:
            url = f"http://{server_address}/system_stats"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_audio_to_remote(self, server_address, audio_data):
        try:
            waveform = audio_data.get('waveform')
            sample_rate = audio_data.get('sample_rate', 44100)
            
            if waveform is None:
                return None
            
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() != 2:
                return None
            
            if waveform.dim() != 2:
                return None
            
            import tempfile
            import uuid
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            import torchaudio
            torchaudio.save(tmp_path, waveform.cpu(), sample_rate)
            
            unique_filename = f"audio_{uuid.uuid4().hex[:8]}.wav"
            
            try:
                with open(tmp_path, 'rb') as f:
                    files = {'image': (unique_filename, f, 'audio/wav')}
                    data = {'overwrite': 'true', 'type': 'input'}
                    
                    url = f"http://{server_address}/upload/image"
                    response = requests.post(url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    returned_name = result.get('name', unique_filename)
                    subfolder = result.get('subfolder', '')
                    
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    if subfolder:
                        return f"{subfolder}/{returned_name}"
                    else:
                        return returned_name
            except:
                pass
            
            try:
                with open(tmp_path, 'rb') as f:
                    files = {'audio': (unique_filename, f, 'audio/wav')}
                    data = {'overwrite': 'true'}
                    
                    url = f"http://{server_address}/upload/audio"
                    response = requests.post(url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    return unique_filename
            except:
                pass
            
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return None
            
        except Exception as e:
            return None
    
    def upload_image_to_remote(self, server_address, image_tensor):
        try:
            if image_tensor.dim() == 4:
                img_array = image_tensor[0]
            elif image_tensor.dim() == 3:
                img_array = image_tensor
            else:
                return None

            img_np = (img_array.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='RGB')

            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

            filename = f"input_{uuid.uuid4().hex[:8]}.png"
            files = {'image': (filename, img_bytes, 'image/png')}
            data = {'overwrite': 'true', 'type': 'input'}

            url = f"http://{server_address}/upload/image"
            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name', filename)
                subfolder = result.get('subfolder', '')
                return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
            else:
                print(f"[remote-nodes] image upload failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"[remote-nodes] image upload error: {e}")
            return None

    def upload_mask_to_remote(self, server_address, mask_tensor):
        try:
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() != 2:
                return None

            mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            # Save as RGB with equal channels so LoadImageMask can extract any channel.
            mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
            img_pil = Image.fromarray(mask_rgb, mode='RGB')

            img_buffer = io.BytesIO()
            img_pil.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

            filename = f"mask_{uuid.uuid4().hex[:8]}.png"
            files = {'image': (filename, img_bytes, 'image/png')}
            data = {'overwrite': 'true', 'type': 'input'}

            url = f"http://{server_address}/upload/image"
            response = requests.post(url, files=files, data=data, timeout=30)

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name', filename)
                subfolder = result.get('subfolder', '')
                return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
            else:
                print(f"[remote-nodes] mask upload failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"[remote-nodes] mask upload error: {e}")
            return None

    def upload_video_to_remote(self, server_address, video_tensor):
        """Upload a frame-sequence tensor (N,H,W,C) as an MP4 to the remote server."""
        try:
            import cv2
            import tempfile

            if video_tensor.dim() == 3:
                video_tensor = video_tensor.unsqueeze(0)
            if video_tensor.dim() != 4:
                return None

            frames = (video_tensor.cpu().numpy() * 255).astype(np.uint8)
            n, h, w, c = frames.shape

            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp_path, fourcc, 24.0, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            filename = f"input_{uuid.uuid4().hex[:8]}.mp4"
            with open(tmp_path, 'rb') as f:
                files = {'image': (filename, f, 'video/mp4')}
                data = {'overwrite': 'true', 'type': 'input'}
                response = requests.post(
                    f"http://{server_address}/upload/image",
                    files=files, data=data, timeout=60
                )

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            if response.status_code == 200:
                result = response.json()
                uploaded_name = result.get('name', filename)
                subfolder = result.get('subfolder', '')
                return f"{subfolder}/{uploaded_name}" if subfolder else uploaded_name
            else:
                print(f"[remote-nodes] video upload failed: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"[remote-nodes] video upload error: {e}")
            return None
    
    def modify_workflow_input(self, workflow, node_id, input_type, input_value, server_address):
        if node_id not in workflow:
            return workflow
        
        node = workflow[node_id]
        
        if input_type == "image":
            if node.get("class_type") == "LoadImage":
                uploaded_filename = self.upload_image_to_remote(server_address, input_value)
                if uploaded_filename:
                    node.setdefault("inputs", {})["image"] = uploaded_filename

        elif input_type == "video":
            if node.get("class_type") == "LoadVideo":
                uploaded_filename = self.upload_video_to_remote(server_address, input_value)
                if uploaded_filename:
                    node.setdefault("inputs", {})["video"] = uploaded_filename
            elif node.get("class_type") == "LoadImage":
                # fallback: send first frame as image
                uploaded_filename = self.upload_image_to_remote(server_address, input_value)
                if uploaded_filename:
                    node.setdefault("inputs", {})["image"] = uploaded_filename
        
        elif input_type == "text":
            if "inputs" not in node:
                node["inputs"] = {}
            
            text_fields = ["text", "prompt", "string", "value"]
            
            found = False
            for field in text_fields:
                if field in node["inputs"]:
                    node["inputs"][field] = str(input_value)
                    found = True
                    break
            
            if not found:
                node["inputs"]["prompt"] = str(input_value)
        
        elif input_type == "mask":
            if node.get("class_type") == "LoadImageMask":
                uploaded_filename = self.upload_mask_to_remote(server_address, input_value)
                if uploaded_filename:
                    node.setdefault("inputs", {})["image"] = uploaded_filename
                    if "channel" not in node["inputs"]:
                        node["inputs"]["channel"] = "red"

        elif input_type == "audio":
            if node.get("class_type") == "LoadAudio":
                uploaded_filename = self.upload_audio_to_remote(server_address, input_value)
                
                if uploaded_filename:
                    if "inputs" not in node:
                        node["inputs"] = {}
                    node["inputs"]["audio"] = uploaded_filename
        
        return workflow
    
    def queue_prompt(self, server_address, workflow):
        try:
            url = f"http://{server_address}/prompt"
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("prompt_id")
            else:
                return None
                
        except Exception as e:
            return None
    
    def wait_for_completion(self, server_address, prompt_id, timeout=600):
        import websocket

        outputs = {}
        done_event = threading.Event()
        error_flag = {"value": False}

        def on_message(ws, message):
            try:
                if isinstance(message, bytes):
                    return

                data = json.loads(message)
                msg_type = data.get("type")
                msg_data = data.get("data", {})

                # Ignore messages from other concurrent prompts.
                msg_prompt_id = msg_data.get("prompt_id")
                if msg_prompt_id and msg_prompt_id != prompt_id:
                    return

                if msg_type == "executed":
                    node_id = msg_data.get("node")
                    output = msg_data.get("output", {})
                    if node_id:
                        outputs[node_id] = output

                elif msg_type == "executing":
                    # node=None signals the workflow finished.
                    if msg_data.get("node") is None:
                        done_event.set()

                elif msg_type == "execution_error":
                    print(f"[remote-nodes] execution error on prompt {prompt_id}: {msg_data}")
                    error_flag["value"] = True
                    done_event.set()

            except Exception as e:
                print(f"[remote-nodes] ws message parse error: {e}")

        def on_error(ws, error):
            print(f"[remote-nodes] ws error: {error}")
            done_event.set()

        def on_close(ws, *args):
            done_event.set()

        ws_url = f"ws://{server_address}/ws?clientId={self.client_id}"
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        finished = done_event.wait(timeout=timeout)
        ws.close()
        ws_thread.join(timeout=2.0)

        if not finished:
            print(f"[remote-nodes] timed out waiting for prompt {prompt_id}")
            return None

        if error_flag["value"]:
            return None

        return outputs if outputs else None
    
    def download_output_file(self, server_address, filename, subfolder="", folder_type="output"):
        try:
            url = f"http://{server_address}/view"
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.gif']:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_path = tmp_file.name
                    
                    try:
                        import cv2
                        cap = cv2.VideoCapture(tmp_path)
                        
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                            frames.append(frame_tensor)
                        
                        cap.release()
                        
                        if frames:
                            video_tensor = torch.stack(frames, dim=0)
                            
                            audio_data = None
                            
                            try:
                                import subprocess
                                
                                audio_tmp_path = tmp_path.replace(file_ext, '.wav')
                                
                                cmd = [
                                    'ffmpeg',
                                    '-i', tmp_path,
                                    '-vn',
                                    '-acodec', 'pcm_s16le',
                                    '-ar', '44100',
                                    '-ac', '2',
                                    '-y',
                                    audio_tmp_path
                                ]
                                
                                result = subprocess.run(
                                    cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    timeout=60
                                )
                                
                                if result.returncode == 0 and os.path.exists(audio_tmp_path):
                                    import torchaudio
                                    waveform, sample_rate = torchaudio.load(audio_tmp_path)
                                    
                                    if waveform.dim() == 2:
                                        waveform = waveform.unsqueeze(0)
                                    
                                    audio_data = {
                                        "waveform": waveform,
                                        "sample_rate": sample_rate
                                    }
                                    
                                    try:
                                        os.unlink(audio_tmp_path)
                                    except:
                                        pass
                                    
                            except:
                                try:
                                    import torchaudio
                                    waveform, sample_rate = torchaudio.load(tmp_path)
                                    
                                    if waveform.dim() == 2:
                                        waveform = waveform.unsqueeze(0)
                                    
                                    audio_data = {
                                        "waveform": waveform,
                                        "sample_rate": sample_rate
                                    }
                                except:
                                    pass
                            
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            
                            return video_tensor, audio_data
                        else:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            return None, None
                            
                    except Exception as video_error:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        return None, None
                
                else:
                    img = Image.open(io.BytesIO(response.content))
                    img = img.convert("RGB")
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    return img_tensor, None
            else:
                return None, None
                
        except Exception as e:
            return None, None
    
    def download_output_audio(self, server_address, filename, subfolder="", folder_type="output"):
        try:
            url = f"http://{server_address}/view"
            params = {
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            }
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                import tempfile
                file_ext = os.path.splitext(filename)[1].lower()
                
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                try:
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(tmp_path)
                    
                    if waveform.dim() == 2:
                        waveform = waveform.unsqueeze(0)
                    
                    audio_data = {
                        "waveform": waveform,
                        "sample_rate": sample_rate
                    }
                    
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    
                    return audio_data
                except Exception as load_error:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    return None
            else:
                return None
                
        except Exception as e:
            return None
    
    def execute_remote(self, remote_host, remote_port, timeout, workflow_file, selected_nodes, saved_state="{}", **kwargs):
        server_address = f"{remote_host}:{remote_port}"
        
        if not self.test_remote_connection(server_address):
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Cannot connect to remote server", self.create_empty_audio(), error_img)

        workflow = self.load_workflow(workflow_file)
        if workflow is None:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Failed to load workflow", self.create_empty_audio(), error_img)

        # Deep copy so cached workflow is never mutated between runs.
        workflow = copy.deepcopy(workflow)

        is_api_format = False

        if isinstance(workflow, dict):
            first_key = next(iter(workflow.keys()), None)
            if first_key and first_key.isdigit():
                is_api_format = True

        if not is_api_format:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Please use API format workflow JSON", self.create_empty_audio(), error_img)

        try:
            selected_map = json.loads(selected_nodes)
        except:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Invalid selected nodes data", self.create_empty_audio(), error_img)

        if not selected_map:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "No nodes selected", self.create_empty_audio(), error_img)
        
        sorted_nodes = sorted(selected_map.items(), key=lambda x: int(x[0]))
        
        type_counters = {"image": 0, "mask": 0, "text": 0, "audio": 0, "video": 0}
        
        for node_id, input_type in sorted_nodes:
            type_counters[input_type] += 1
            counter = type_counters[input_type]
            
            input_key = f"{input_type}_{counter}"
            input_value = kwargs.get(input_key)
            
            if input_value is None:
                continue
            
            workflow = self.modify_workflow_input(workflow, node_id, input_type, input_value, server_address)
        
        prompt_id = self.queue_prompt(server_address, workflow)
        
        if prompt_id is None:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Failed to submit workflow", self.create_empty_audio(), error_img)

        all_outputs = self.wait_for_completion(server_address, prompt_id, timeout=timeout)

        if all_outputs is None:
            error_img = torch.zeros((1, 64, 64, 3))
            return (error_img, "Execution failed or timed out", self.create_empty_audio(), error_img)
        
        output_images = []
        output_texts = []
        output_audios = []
        output_videos = []
        
        if all_outputs:
            save_image_nodes = []
            preview_image_nodes = []
            text_nodes = []
            video_nodes = []
            audio_nodes = []
            
            for node_id, node_output in all_outputs.items():
                if not node_output:
                    continue
                
                node_class = workflow.get(node_id, {}).get("class_type", "")
                
                if "images" in node_output:
                    if node_class == "SaveImage":
                        save_image_nodes.append(node_id)
                    elif node_class == "PreviewImage":
                        preview_image_nodes.append(node_id)
                
                if "text" in node_output or "string" in node_output:
                    text_nodes.append(node_id)
                
                if "gifs" in node_output:
                    video_nodes.append(node_id)
                
                if "audio" in node_output or "audios" in node_output:
                    audio_nodes.append(node_id)
            
            final_image_node = None
            if save_image_nodes:
                final_image_node = max(save_image_nodes, key=lambda x: int(x))
            elif preview_image_nodes:
                final_image_node = max(preview_image_nodes, key=lambda x: int(x))
            else:
                for node_id, node_output in all_outputs.items():
                    if node_output and "images" in node_output:
                        final_image_node = node_id
            
            final_text_node = None
            if text_nodes:
                final_text_node = max(text_nodes, key=lambda x: int(x))
            
            final_video_node = None
            if video_nodes:
                final_video_node = max(video_nodes, key=lambda x: int(x))
            
            final_audio_node = None
            if audio_nodes:
                final_audio_node = max(audio_nodes, key=lambda x: int(x))
            
            if final_image_node and final_image_node in all_outputs:
                node_output = all_outputs[final_image_node]
                if node_output:
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img_tensor, _ = self.download_output_file(
                                server_address,
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output")
                            )
                            if img_tensor is not None:
                                output_images.append(img_tensor)
            
            if final_text_node and final_text_node in all_outputs:
                node_output = all_outputs[final_text_node]
                if node_output:
                    if "text" in node_output:
                        if isinstance(node_output["text"], list):
                            for text_item in node_output["text"]:
                                output_texts.append(str(text_item))
                        else:
                            output_texts.append(str(node_output["text"]))
                    
                    if "string" in node_output:
                        if isinstance(node_output["string"], list):
                            for string_item in node_output["string"]:
                                output_texts.append(str(string_item))
                        else:
                            output_texts.append(str(node_output["string"]))
            
            if final_video_node and final_video_node in all_outputs:
                node_output = all_outputs[final_video_node]
                if node_output:
                    if "gifs" in node_output:
                        for gif_info in node_output["gifs"]:
                            video_filename = gif_info.get("filename", "")
                            
                            video_tensor, video_audio = self.download_output_file(
                                server_address,
                                video_filename,
                                gif_info.get("subfolder", ""),
                                gif_info.get("type", "output")
                            )
                            
                            if video_tensor is not None:
                                output_videos.append(video_tensor)
                                
                                if video_audio is not None:
                                    output_audios.append(video_audio)
            
            if final_audio_node and final_audio_node in all_outputs:
                node_output = all_outputs[final_audio_node]
                if node_output:
                    if "audio" in node_output:
                        audio_list = node_output["audio"]
                        if not isinstance(audio_list, list):
                            audio_list = [audio_list]
                        
                        for audio_item in audio_list:
                            if isinstance(audio_item, dict) and "filename" in audio_item:
                                audio_file = audio_item.get("filename", "")
                                
                                audio_data = self.download_output_audio(
                                    server_address,
                                    audio_file,
                                    audio_item.get("subfolder", ""),
                                    audio_item.get("type", "output")
                                )
                                
                                if audio_data is not None:
                                    output_audios.append(audio_data)
                    
                    elif "audios" in node_output:
                        audios_list = node_output["audios"]
                        if not isinstance(audios_list, list):
                            audios_list = [audios_list]
                        
                        for audio_item in audios_list:
                            if isinstance(audio_item, dict) and "filename" in audio_item:
                                audio_file = audio_item.get("filename", "")
                                
                                audio_data = self.download_output_audio(
                                    server_address,
                                    audio_file,
                                    audio_item.get("subfolder", ""),
                                    audio_item.get("type", "output")
                                )
                                
                                if audio_data is not None:
                                    output_audios.append(audio_data)
        
        # Concatenate batch tensors so multi-image/video outputs are preserved.
        final_image = torch.cat(output_images, dim=0) if output_images else torch.zeros((1, 64, 64, 3))
        final_text = "\n".join(output_texts) if output_texts else "Execution successful"
        final_audio = output_audios[-1] if output_audios else self.create_empty_audio()
        final_video = torch.cat(output_videos, dim=0) if output_videos else torch.zeros((1, 64, 64, 3))
        
        return (final_image, final_text, final_audio, final_video)


NODE_CLASS_MAPPINGS = {
    "RemoteWorkflowExecutor": RemoteWorkflowExecutor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteWorkflowExecutor": "Remote Workflow Executor"
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
