# ComfyUI-AstrBubble-Nodes

AstrBubble nodes for `astrbot_plugin_comfyui_bubble`.

## Install

Clone or copy this repository into:

```text
ComfyUI/custom_nodes/ComfyUI-AstrBubble-Nodes
```

Restart ComfyUI after installation.

No extra Python packages are required beyond the standard ComfyUI runtime.

## Nodes

Input nodes:

- `AstrBubble_TextInput`: receives text injected by the AstrBot plugin.
- `AstrBubble_ImageInput`: receives base64 image data injected by the AstrBot plugin and outputs `IMAGE`.
- `AstrBubble_VideoInput`: receives a server-side video filename/path injected by the AstrBot plugin and outputs `STRING`.

Output nodes:

- `AstrBubble_TextOutput`: writes text into ComfyUI history.
- `AstrBubble_ImageOutput`: saves images into ComfyUI output history.
- `AstrBubble_VideoOutput`: copies a video file into ComfyUI output history.

## Slot Rules

Every AstrBubble node has:

- `index`: a positive 1-based slot number.
- `explain`: a human-readable explanation, such as `positive prompt`, `negative prompt`, `reference image`, or `first frame`.

The AstrBot plugin scans workflow API JSON files and requires each same-direction, same-type group to be consecutively indexed from `1`.

Examples:

- Two text inputs: `TextInput index=1 explain=positive prompt`, `TextInput index=2 explain=negative prompt`.
- Two image inputs: `ImageInput index=1 explain=source image`, `ImageInput index=2 explain=style reference`.
- One image output: `ImageOutput index=1 explain=result image`.

The plugin injects by `index`, not by node creation order or JSON order. The `class_type`
names in `nodes.py` are the public protocol and should not be renamed.

## Notes

- Export workflows with ComfyUI's API workflow export, then upload the JSON to `astrbot_plugin_comfyui_bubble`.
- `AstrBubble_VideoInput` outputs a string path/name. Connect it to downstream nodes that accept a video path or filename.
- `AstrBubble_VideoOutput` expects a string path/name and copies that file into ComfyUI's output directory so AstrBot can find it from history.
# ComfyUI_AstrBot_Bubble

