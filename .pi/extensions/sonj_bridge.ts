import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  const baseUrl = () => (process.env.ME3_ENGINE_URL || "http://127.0.0.1:8766").replace(/\/+$/, "");
  const bridgeMode = process.env.ME3_BRIDGE_MODE || "front";
  const agentRole = process.env.SONJ_ROLE || bridgeMode;

  async function postJson(path: string, body: Record<string, unknown>) {
    const res = await fetch(`${baseUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await res.text();
    try {
      return { ok: res.ok, status: res.status, data: JSON.parse(text) };
    } catch {
      return { ok: res.ok, status: res.status, data: { raw: text } };
    }
  }

  async function getJson(path: string) {
    const res = await fetch(`${baseUrl()}${path}`);
    const text = await res.text();
    try {
      return { ok: res.ok, status: res.status, data: JSON.parse(text) };
    } catch {
      return { ok: res.ok, status: res.status, data: { raw: text } };
    }
  }

  // Front + Back get device_tool (Front restricted strictly to <1s actions).
  if (bridgeMode !== "doctor") {
    pi.registerTool({
      name: "device_tool",
      label: "Device Tool",
      description:
        bridgeMode === "front"
          ? "Execute ONLY <1s local device actions (get_time, get_weather, calculate, set_timer, set_alarm, set_lights, adjust_lights, adjust_temperature, adjust_volume, media_control, set_reminder, list_reminders, make_call, silence, remember, recall, forget, open_url, open_app). NEVER use for web search, files, code, or long tasks — delegate those to role='back' or 'doctor'."
          : "Execute a local device/assistant/web tool via SONJ4 (get_time, get_weather, calculate, set_timer, set_alarm, set_lights, adjust_lights, adjust_temperature, adjust_volume, media_control, set_reminder, list_reminders, make_call, silence, remember, recall, forget, web_search, read_page, browse_page, open_url, open_app).",
      parameters: Type.Object({
        name: Type.String({
          description:
            bridgeMode === "front"
              ? "Quick tool name only: get_time, get_weather, calculate, set_timer, set_alarm, set_lights, adjust_lights, adjust_temperature, adjust_volume, media_control, set_reminder, list_reminders, make_call, silence, remember, recall, forget, open_url, open_app"
              : "Tool name, e.g. get_time, set_lights, web_search, browse_page",
        }),
        arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Tool JSON arguments" })),
      }),
      async execute(_id, params) {
        const slowOrWorkerTools = new Set([
          "web_search", "read_page", "browse_page", "read_file", "write_file",
          "list_files", "run_code", "tree", "cd", "pwd", "call_api", "register_api",
          "audio_overview", "send_email",
        ]);
        if (bridgeMode === "front" && slowOrWorkerTools.has(String(params.name || "").trim())) {
          return {
            content: [{
              type: "text",
              text: `Latency boundary: '${params.name}' is forbidden on Front. Delegate immediately using delegate_task(brief, role='back') instead.`,
            }],
            details: { blocked_on_front: true, tool: params.name },
          };
        }
        const r = await postJson("/api/tool", {
          name: params.name,
          arguments: params.arguments || {},
        });
        const text = (r.data as any)?.result ?? (r.data as any)?.error ?? JSON.stringify(r.data);
        return {
          content: [{ type: "text", text: String(text) }],
          details: r.data,
        };
      },
    });
  }

  // Back workers get full MCP desktop/browser automation + peer coordination.
  if (bridgeMode === "back") {
    pi.registerTool({
      name: "mcp_call",
      label: "MCP Tool Call",
      description:
        "Call a tool on a connected MCP server (computer-mcp for desktop control, browser-mcp for web automation).",
      parameters: Type.Object({
        server: Type.String({ description: "MCP server name: computer-mcp or browser-mcp" }),
        tool: Type.String({ description: "Tool name on that server" }),
        arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Tool JSON arguments" })),
      }),
      async execute(_id, params) {
        const r = await postJson("/api/mcp/call", {
          server: params.server,
          tool: params.tool,
          arguments: params.arguments || {},
        });
        const data: any = r.data;
        let text = "";
        if (data?.ok) {
          const content = data?.result?.content;
          text = Array.isArray(content)
            ? content.map((c: any) => c?.text ?? "").join("\n")
            : JSON.stringify(data?.result ?? {});
        } else {
          text = data?.error ?? JSON.stringify(data);
        }
        return { content: [{ type: "text", text }], details: data };
      },
    });
  }

  // Shared agent-to-agent coordination tools (Front, Back workers, Doctor)
  if (bridgeMode === "back" || bridgeMode === "doctor") {
    pi.registerTool({
      name: "post_community",
      label: "Post to Community",
      description: "Post a status or coordination message into community · agents talking (independent of Front).",
      parameters: Type.Object({
        message: Type.String({ description: "Short clear coordination message" }),
        to_role: Type.Optional(Type.String({ description: "Target role: front, doctor, back-1..back-5, or all" })),
      }),
      async execute(_id, params) {
        const r = await postJson("/api/community", {
          from_role: agentRole,
          to_role: params.to_role || "front",
          message: params.message,
        });
        return { content: [{ type: "text", text: JSON.stringify(r.data) }], details: r.data };
      },
    });
    pi.registerTool({
      name: "ask_agent",
      label: "Ask Peer Agent",
      description: "Consult another agent (doctor or an idle back-1..back-5 peer) directly and record it in community · agents talking.",
      parameters: Type.Object({
        role: Type.String({ description: "Target role: doctor or back / back-1..back-5" }),
        question: Type.String({ description: "Clear, concise question" }),
        context: Type.Optional(Type.String({ description: "Optional context" })),
      }),
      async execute(_id, params) {
        const r = await postJson("/api/ask_agent", {
          role: params.role,
          question: params.question,
          context: params.context || "",
          from_role: agentRole,
        });
        const reply = (r.data as any)?.reply ?? (r.data as any)?.error ?? JSON.stringify(r.data);
        return { content: [{ type: "text", text: String(reply) }], details: r.data };
      },
    });
  }

  if (bridgeMode === "doctor") {
    pi.registerTool({
      name: "doctor_app_control",
      label: "Doctor App Control",
      description:
        "Inspect SONJ4 internals (inspect_state, settings_catalog, inspect_architecture, inspect_tokens, inspect_logs, performance_review, system_health, feature_distribution, introspection, app_guide) OR propose a setting/settings/model/restart change (opens 15s auto-confirm pop-up; no self-coding).",
      parameters: Type.Object({
        action: Type.String({
          description:
            "Read actions: inspect_state | settings_catalog | inspect_architecture | inspect_tokens | inspect_logs | performance_review | system_health | feature_distribution | introspection | app_guide. Mutating proposals (15s auto-confirm pop-up): propose_setting | propose_settings | propose_role_model | propose_restart | propose_reset_session.",
        }),
        key: Type.Optional(Type.String({ description: "Setting name or role name" })),
        value: Type.Optional(Type.Any({ description: "New value for setting, model id, or dict of {key:value} for propose_settings" })),
        summary: Type.Optional(Type.String({ description: "Human summary of proposed change" })),
      }),
      async execute(_id, params) {
        const r = await postJson("/api/doctor/control", {
          action: params.action,
          key: params.key ?? "",
          value: params.value,
          summary: params.summary ?? "",
        });
        return {
          content: [{ type: "text", text: JSON.stringify(r.data, null, 2) }],
          details: r.data,
        };
      },
    });
    return;
  }

  if (bridgeMode !== "front") return;

  pi.registerTool({
    name: "list_agents",
    label: "List Agents",
    description: "List the SONJ4 roster (front, doctor, classifier, back-1..back-5).",
    parameters: Type.Object({}),
    async execute() {
      const r = await getJson("/api/roles");
      const roles = (r.data && typeof r.data === "object" && "roles" in r.data ? (r.data as any).roles : {}) || {};
      const lines = Object.entries(roles).map(([role, cfg]) => `${role}: ${(cfg as any)?.model || "?"}`);
      return {
        content: [{ type: "text", text: lines.length ? lines.join("\n") : "no roles configured" }],
        details: roles,
      };
    },
  });

  pi.registerTool({
    name: "classify_decision",
    label: "Decision Classifier (<100ms, non-LLM)",
    description:
      "Call the non-LLM Decision Classifier (<100ms, +10% Front bias) to get a structured routing choice (front | back | doctor) without using an LLM.",
    parameters: Type.Object({
      task: Type.String({ description: "User request or task text to classify" }),
    }),
    async execute(_id, params) {
      const r = await postJson("/api/classify", { text: params.task });
      return {
        content: [{ type: "text", text: JSON.stringify(r.data) }],
        details: r.data,
      };
    },
  });

  pi.registerTool({
    name: "delegate_task",
    label: "Delegate Task",
    description:
      "Hand a substantial task (build, web research, verify/audit, summarize, automate, or diagnostics) to an idle Back worker (back-1..back-5) or doctor to run in the BACKGROUND. Returns immediately; result reports into chat + TTS when done.",
    parameters: Type.Object({
      brief: Type.String({ description: "Clear, complete task brief including output paths or mode (build/research/verify/summarize)" }),
      role: Type.Optional(Type.String({ description: "Target role: back (auto-picks idle back-1..back-5) or doctor" })),
    }),
    async execute(_id, params) {
      const r = await postJson("/api/delegate", { brief: params.brief, role: params.role || "back" });
      const msg = (r.data as any)?.message ?? (r.data as any)?.error ?? JSON.stringify(r.data);
      return {
        content: [{ type: "text", text: String(msg) }],
        details: r.data,
      };
    },
  });

  pi.registerTool({
    name: "ask_agent",
    label: "Ask Agent",
    description:
      "SYNCHRONOUS consult: ask doctor or an idle back worker (back / back-1..back-5) and wait for its answer. Prefer delegate_task for >2s work.",
    parameters: Type.Object({
      role: Type.String({ description: "Target role: doctor or back / back-1..back-5" }),
      question: Type.String({ description: "Clear, short question" }),
      context: Type.Optional(Type.String({ description: "Optional background context from Front" })),
    }),
    async execute(_id, params) {
      const r = await postJson("/api/ask_agent", {
        role: params.role,
        question: params.question,
        context: params.context || "",
        from_role: "front",
      });
      const reply = (r.data as any)?.reply ?? (r.data as any)?.error ?? JSON.stringify(r.data);
      return {
        content: [{ type: "text", text: String(reply) }],
        details: r.data,
      };
    },
  });

  pi.registerTool({
    name: "start_crew",
    label: "Start Crew",
    description: "Start a multi-agent background crew across Back workers (e.g. ['back-1','back-2']) and/or doctor.",
    parameters: Type.Object({
      brief: Type.String({ description: "Task brief for the crew" }),
      roles: Type.Array(Type.String(), { description: "Ordered role names, e.g. ['back-1','back-2'] or ['back','back']" }),
      process: Type.Optional(Type.String({ description: "sequential or hierarchical" })),
    }),
    async execute(_id, params) {
      const r = await postJson("/api/crew", {
        brief: params.brief,
        roles: params.roles,
        process: params.process || "sequential",
      });
      const msg = (r.data as any)?.message ?? JSON.stringify(r.data);
      return {
        content: [{ type: "text", text: String(msg) }],
        details: r.data,
      };
    },
  });
}
