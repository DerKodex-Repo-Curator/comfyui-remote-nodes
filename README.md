# comfyui-remote-nodes

A ComfyUI custom node that lets you execute workflows on a **remote ComfyUI server** from your local instance. Send images, text, audio, or video as inputs and receive outputs back — all through a single node.

## Features

- **Remote execution** — connect to any ComfyUI server on your network, submit workflows, and retrieve results
- **Multi-type I/O** — supports image, mask, text, audio, and video inputs and outputs
- **Workflow parser** — visual dialog to load an API-format JSON, inspect its nodes, and select which ones receive local inputs
- **Dynamic ports** — input connectors are generated automatically based on your node selection
- **IP privacy** — the server address is masked by default (useful during screen sharing)
- **WebSocket** — real-time monitoring of remote execution status

## Installation

1. Copy this folder to your ComfyUI `custom_nodes` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## Usage

### 1. Add the node

Search for **`Remote Workflow Executor`** under the `remote_nodes` category.

### 2. Configure the remote server

Click the ⚙️ button on the node and enter the remote server IP and port (default `8188`).

### 3. Load a workflow

1. Click **🔧 Parse Workflow**.
2. In the dialog, upload the workflow's **API format JSON** (exported from ComfyUI via *Save (API Format)*).
3. The parser lists all detected input and output nodes.

### 4. Select input nodes

Toggle on the nodes you want to feed data into from your local graph, then click **💾 Save & Update Ports**.

### 5. Connect and run

The node generates typed input connectors (`image_1`, `text_1`, etc.). Wire them up and run your workflow — the node uploads inputs, executes the remote workflow, then downloads and returns the outputs.

## Ports

### Inputs

| Port | Type | Description |
|------|------|-------------|
| `image_N` | IMAGE | Replaces a remote `LoadImage` node |
| `mask_N` | MASK | Replaces a remote `LoadImageMask` node |
| `text_N` | STRING | Replaces a remote text node (`prompt` / `text` field) |
| `audio_N` | AUDIO | Replaces a remote `LoadAudio` node |
| `video_N` | IMAGE | Replaces a remote `LoadVideo` node (frame sequence) |

### Outputs

| Port | Type | Description |
|------|------|-------------|
| `output_image` | IMAGE | Image result from the remote workflow |
| `output_text` | STRING | Text result from the remote workflow |
| `output_audio` | AUDIO | Audio result from the remote workflow |
| `output_video` | IMAGE | Video result from the remote workflow (frame sequence) |

## Supported remote node types

**Input nodes** — `LoadImage`, `LoadImageMask`, `LoadVideo`, `LoadAudio`, `CR Prompt Text`, `Text`, `easy showAnything`

**Output nodes** — `SaveImage`, `PreviewImage`, `VHS_VideoCombine`, `SaveAudio`, `easy showAnything`

## Notes

- The remote server must be reachable on both HTTP and WebSocket ports.
- Workflows must be in **API format** JSON (not the default save format).
- Default execution timeout is **600 seconds**.
- Large files (video) may take time to transfer.
- The remote server must have all required custom nodes and models installed.

## License

MIT
