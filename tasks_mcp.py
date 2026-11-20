"""ME3 task/device MCP server: the agent's hands, on the device.

This exposes the same tool implementations the laptop app uses (toolbox.py), so
there is one source of truth for the device commands. The vocabulary follows the
ME2 label set: lights on/off/dim/bright, timer, alarm, thermostat, media
play/pause/next/previous, volume, reminders, call, search and silence.

On the Raspberry Pi the simulated device state inside toolbox.py is where real
GPIO or service calls belong.

Run standalone (stdio, speaks MCP):
    python me3_laptop/tasks_mcp.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mcp_core import Server, ToolError, make_tool
from toolbox import ToolBox

# The ME2 intents this server covers, for anyone auditing the mapping.
ME2_INTENTS: dict[str, list[str]] = {
    "Play Music": ["music_play"],
    "Ask a Question / Search": ["search", "get_time", "get_weather", "web_search", "read_page"],
    "Control Lights": ["lights_on", "lights_off"],
    "Dim Lights": ["lights_dim", "lights_bright"],
    "Set Timer": ["timer_set"],
    "Set Alarm": ["alarm_set"],
    "Adjust Thermostat": ["temp_up", "temp_down"],
    "Media Control": ["media_play", "media_pause", "media_next", "media_prev", "volume_up", "volume_down"],
    "Reminders and Lists": ["reminder_set", "reminder_check"],
    "Calls and Messaging": ["call"],
    "Silence": ["silence"],
}

MAP_TO_TOOLS: dict[str, str] = {
    "lights_on": "set_lights(state=on)",
    "lights_off": "set_lights(state=off)",
    "lights_dim": "adjust_lights(direction=dim)",
    "lights_bright": "adjust_lights(direction=brighten)",
    "timer_set": "set_timer(seconds, label)",
    "alarm_set": "set_alarm(time, label)",
    "temp_up": "adjust_temperature(direction=warmer)",
    "temp_down": "adjust_temperature(direction=cooler)",
    "media_play": "media_control(action=play)",
    "media_pause": "media_control(action=pause)",
    "media_next": "media_control(action=next)",
    "media_prev": "media_control(action=previous)",
    "volume_up": "adjust_volume(direction=up)",
    "volume_down": "adjust_volume(direction=down)",
    "reminder_set": "set_reminder(text, when)",
    "reminder_check": "list_reminders()",
    "call": "make_call(contact)",
    "silence": "silence()",
    "music_play": "media_control(action=play) after a music search",
}


def _handler(box: ToolBox, name: str) -> Callable[..., str]:
    """Adapt one ToolBox tool to the MCP handler contract.

    ToolBox reports failure in its result object; MCP reports it with isError, so
    a failed tool raises here and the core turns it into an error result.
    """

    def call(**arguments: Any) -> str:
        result = box.call(name, arguments)
        if not result.ok:
            # ToolBox already prefixes its message with the tool name; strip it so
            # the MCP error does not read "set_lights failed: set_lights failed:".
            message = result.content.removeprefix(f"{name} failed: ")
            raise ToolError(message)
        return result.content

    call.__name__ = name
    return call


def build_server(box: ToolBox | None = None) -> Server:
    toolbox = box or ToolBox()
    server = Server("me3-tasks", version="0.1.0")

    for schema in toolbox.schemas():
        function = schema.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        parameters = function.get("parameters") or {}
        server.add(
            make_tool(
                name,
                str(function.get("description") or ""),
                parameters.get("properties") or {},
                _handler(toolbox, name),
                parameters.get("required") or [],
            )
        )

    server.add(make_tool("tasks_info", "List the device commands and the ME2 intents they cover.", {}, lambda: _info(toolbox)))
    return server


def _info(box: ToolBox) -> str:
    """Tool list plus the ME2 intent mapping."""
    names = [str((schema.get("function") or {}).get("name")) for schema in box.schemas()]
    lines = ["device commands: " + ", ".join(sorted(names))]
    lines.append("")
    lines.append("ME2 intent coverage:")
    for intent, labels in ME2_INTENTS.items():
        lines.append(f"  {intent}: {', '.join(labels)}")
    lines.append("")
    lines.append("label to tool mapping:")
    for label, tool in MAP_TO_TOOLS.items():
        lines.append(f"  {label} -> {tool}")
    return chr(10).join(lines)


if __name__ == "__main__":
    raise SystemExit(build_server().serve())
