"use client";

import { Bot } from "lucide-react";
import type { Agent, AgentEvent, GuardrailCheck } from "@/lib/types";
import { AgentsList } from "./agents-list";
import { Guardrails } from "./guardrails";
import { ConversationContext } from "./conversation-context";
import { RunnerOutput } from "./runner-output";
import { useState } from "react";

interface AgentPanelProps {
  agents: Agent[];
  currentAgent: string;
  events: AgentEvent[];
  guardrails: GuardrailCheck[];
  context: {
    passenger_name?: string;
    confirmation_number?: string;
    seat_number?: string;
    flight_number?: string;
    account_number?: string;
  };
}

export function AgentPanel({
  agents,
  currentAgent,
  events,
  guardrails,
  context,
}: AgentPanelProps) {
  const activeAgent = agents.find((a) => a.name === currentAgent);
  const runnerEvents = events.filter((e) => e.type !== "message");

  // Estado para upload de PDF/RAG
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  async function handleRagUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/rag/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.success) {
        setUploadStatus(`Arquivo "${file.name}" enviado e indexado para RAG!`);
      } else {
        setUploadStatus(`Erro ao processar: ${data.error || "desconhecido"}`);
      }
    } catch (err) {
      setUploadStatus("Erro de rede ao enviar arquivo.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="w-3/5 h-full flex flex-col border-r border-gray-200 bg-white rounded-xl shadow-sm">
      <div className="bg-blue-600 text-white h-12 px-4 flex items-center gap-3 shadow-sm rounded-t-xl">
        <Bot className="h-5 w-5" />
        <h1 className="font-semibold text-sm sm:text-base lg:text-lg">Agent View</h1>
        <span className="ml-auto text-xs font-light tracking-wide opacity-80">
          Airline&nbsp;Co.
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-gray-50/50">
      {/* Campo de upload de PDF para RAG */}
        <div className="mb-4">
          <label className="block text-xs font-semibold mb-1">
            Upload de PDF para Portfolio RAG:
          </label>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleRagUpload}
            disabled={uploading}
            className="block text-xs"
          />
          {uploadStatus && (
            <div className="text-xs mt-1 text-blue-700">{uploadStatus}</div>
          )}
        </div>
        <AgentsList agents={agents} currentAgent={currentAgent} />
        <Guardrails
          guardrails={guardrails}
          inputGuardrails={activeAgent?.input_guardrails ?? []}
        />
        <ConversationContext context={context} />
        <RunnerOutput runnerEvents={runnerEvents} />
      </div>
    </div>
  );
}