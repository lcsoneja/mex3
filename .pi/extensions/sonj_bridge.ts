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

  // Front + Back get full device_tool access (including web, file, and code tools when Front acts independently).
  if (bridgeMode !== "doctor") {
    pi.registerTool({
      name: "device_tool",
      label: "Device Tool",
      description:
        "Execute any local SONJ4 device, assistant, file, code, or web tool (get_time, get_weather, get_stock_price, calculate, set_timer, set_alarm, set_lights, adjust_lights, adjust_temperature, adjust_volume, media_control, set_reminder, list_reminders, make_call, silence, remember, recall, forget, web_search, read_page, browse_page, read_file, write_file, list_files, run_code, tree, cd, pwd, call_api, register_api, open_url, open_app, send_email).",
      parameters: Type.Object({
        name: Type.String({
          description:
            "SONJ4 tool name, e.g. get_time, get_stock_price, set_lights, web_search, browse_page, read_file, write_file, run_code",
        }),
        arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Tool JSON arguments" })),
      }),
      async execute(_id, params) {
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

  // Front gets standard SONJ4 MCP servers; ONLY Back agents get github-mcp.
  if (bridgeMode === "back" || bridgeMode === "front") {
    const mcpDesc =
      bridgeMode === "back"
        ? "Call a tool on a connected SONJ4 MCP server (asr-mcp, tts-mcp, tasks-mcp, browser-mcp, computer-mcp, agents-mcp, github-mcp). Use github-mcp for GitHub API tasks (repo_overview, create_repo, get_repo, list_repos, get_file, create_or_update_file, edit_file, delete_file, create_branch, list_commits, create_issue, create_pull_request, api_request)."
        : "Call a tool on a connected SONJ4 MCP server (asr-mcp, tts-mcp, tasks-mcp, browser-mcp, computer-mcp, agents-mcp). Note: github-mcp is Back-only; delegate GitHub API work to role='back'.";
    const serverDesc =
      bridgeMode === "back"
        ? "MCP server name: asr-mcp, tts-mcp, tasks-mcp, browser-mcp, computer-mcp, agents-mcp, or github-mcp"
        : "MCP server name: asr-mcp, tts-mcp, tasks-mcp, browser-mcp, computer-mcp, or agents-mcp (github-mcp is Back-only)";
    pi.registerTool({
      name: "mcp_call",
      label: "MCP Tool Call",
      description: mcpDesc,
      parameters: Type.Object({
        server: Type.String({ description: serverDesc }),
        tool: Type.String({ description: "Tool name on that server" }),
        arguments: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Tool JSON arguments" })),
      }),
      async execute(_id, params) {
        if (bridgeMode !== "back" && String(params.server || "").trim() === "github-mcp") {
          return {
            content: [{ type: "text", text: "github-mcp is restricted to Back agents (back-1..back-5). Delegate GitHub tasks via delegate_task(brief, role='back')." }],
            details: { ok: false, error: "github-mcp is Back-only" },
          };
        }
        const r = await postJson("/api/mcp/call", {
          server: params.server,
          tool: params.tool,
          arguments: params.arguments || {},
          from_role: agentRole,
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
  if (bridgeMode === "front" || bridgeMode === "back" || bridgeMode === "doctor") {
    pi.registerTool({
      name: "post_community",
      label: "Post to Community",
      description: "Post a status or coordination message into community · agents talking.",
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
  }
  if (bridgeMode === "back" || bridgeMode === "doctor") {
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

  if (bridgeMode === "doctor" || bridgeMode === "front") {
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
  }
  if (bridgeMode === "doctor") {
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
