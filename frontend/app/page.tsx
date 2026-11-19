import { AssistPanel } from "@/components/AssistPanel";
import { DiagnosisEditor } from "@/components/DiagnosisEditor";
import { ImportBar } from "@/components/ImportBar";
import { RiskInspector } from "@/components/RiskInspector";
import { TermPanel } from "@/components/TermPanel";

export default function WorkbenchPage() {
  return (
    <div className="shell">
      <ImportBar />
      <main className="workbench">
        <TermPanel />
        <section className="panel panel-center">
          <DiagnosisEditor />
        </section>
        <aside className="panel panel-right">
          <RiskInspector />
          <AssistPanel />
        </aside>
      </main>
    </div>
  );
}
