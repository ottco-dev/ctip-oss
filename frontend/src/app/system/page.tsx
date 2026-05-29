"use client";

import React, { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Monitor, Activity, SlidersHorizontal } from "lucide-react";
import { SystemStatusTab } from "@/components/system/SystemStatusTab";
import { ProcessesTab } from "@/components/system/ProcessesTab";
import { SetupTab } from "@/components/system/SetupTab";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SystemTab = "status" | "processes" | "setup";

const TABS: { id: SystemTab; label: string; icon: React.ElementType }[] = [
  { id: "status",    label: "Status",    icon: Monitor },
  { id: "processes", label: "Processes", icon: Activity },
  { id: "setup",     label: "Setup",     icon: SlidersHorizontal },
];

// ---------------------------------------------------------------------------
// Inner component (uses useSearchParams — must be inside Suspense)
// ---------------------------------------------------------------------------

function SystemPageInner() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const rawTab = searchParams.get("tab") as SystemTab | null;
  const activeTab: SystemTab =
    rawTab && TABS.some((t) => t.id === rawTab) ? rawTab : "status";

  const handleTabChange = (tab: SystemTab) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", tab);
    router.replace(`/system?${params.toString()}`);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div
        className="flex items-center gap-1 px-4 py-2 shrink-0"
        style={{ borderBottom: "1px solid #21262d", background: "#161b22" }}
      >
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => handleTabChange(id)}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-md text-xs font-medium transition-colors",
              activeTab === id
                ? "bg-[#21262d] text-white"
                : "text-[#8b949e] hover:text-white hover:bg-[#21262d]/50"
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden min-h-0">
        {activeTab === "status"    && <SystemStatusTab />}
        {activeTab === "processes" && <ProcessesTab />}
        {activeTab === "setup"     && <SetupTab />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page export — wraps inner in Suspense for useSearchParams
// ---------------------------------------------------------------------------

export default function SystemPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full">
          <div className="w-5 h-5 rounded-full border-2 border-t-transparent border-blue-400 animate-spin" />
        </div>
      }
    >
      <SystemPageInner />
    </Suspense>
  );
}
