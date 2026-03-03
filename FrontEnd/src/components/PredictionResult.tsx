import { Shield, ShieldAlert, ShieldCheck, Scan, Activity, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PredictResponse, ShapTopToken } from "@/lib/mockPredict";
import { getVerdictView, isShapExplanation, isShapTopToken } from "@/lib/mockPredict";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface PredictionResultProps {
  result: PredictResponse | null;
  isLoading: boolean;
}

const PredictionResult = ({ result, isLoading }: PredictionResultProps) => {
  if (isLoading) {
    return (
      <div className="relative rounded-2xl glass p-8 overflow-hidden border-glow">
        {/* Animated scan lines */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute inset-x-0 h-32 bg-gradient-to-b from-primary/30 via-primary/10 to-transparent scan-line" />
        </div>

        {/* Corner decorations */}
        <div className="absolute top-3 left-3 w-6 h-6 border-t border-l border-primary/60" />
        <div className="absolute top-3 right-3 w-6 h-6 border-t border-r border-primary/60" />
        <div className="absolute bottom-3 left-3 w-6 h-6 border-b border-l border-primary/60" />
        <div className="absolute bottom-3 right-3 w-6 h-6 border-b border-r border-primary/60" />

        <div className="relative flex flex-col items-center justify-center gap-5 min-h-[280px]">
          {/* Animated icon */}
          <div className="relative">
            <div className="absolute inset-0 rounded-full bg-primary/20 pulse-glow" />
            <div className="relative p-6 rounded-full bg-primary/10 border border-primary/30">
              <Scan className="w-12 h-12 text-primary animate-pulse" />
            </div>
          </div>

          <div className="text-center space-y-2">
            <p className="text-xl font-semibold text-foreground flex items-center gap-2 justify-center">
              <Activity className="w-5 h-5 text-primary" />
              Scanning Code
              <span className="inline-flex">
                <span className="animate-bounce" style={{ animationDelay: "0ms" }}>.</span>
                <span className="animate-bounce" style={{ animationDelay: "150ms" }}>.</span>
                <span className="animate-bounce" style={{ animationDelay: "300ms" }}>.</span>
              </span>
            </p>
            <p className="text-sm text-muted-foreground">
              Running GraphCodeBERT neural analysis
            </p>
          </div>

          {/* Progress indicator */}
          <div className="w-full max-w-xs">
            <div className="h-1.5 bg-muted/50 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-primary via-primary/60 to-primary rounded-full shimmer" style={{ width: "100%" }} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="relative rounded-2xl border border-dashed border-border/60 bg-card/30 p-8 min-h-[340px]">
        {/* Grid pattern overlay */}
        <div className="absolute inset-0 grid-bg opacity-30 rounded-2xl" />

        <div className="relative flex flex-col items-center justify-center gap-5 h-full">
          <div className="p-6 rounded-full bg-muted/30 border border-border/50">
            <Shield className="w-12 h-12 text-muted-foreground/60" />
          </div>
          <div className="text-center space-y-2">
            <p className="text-lg font-medium text-muted-foreground">Awaiting Analysis</p>
            <p className="text-sm text-muted-foreground/70 max-w-[200px]">
              Enter your C/C++ code and click analyze to begin
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground/50 font-mono">
            <span className="w-2 h-2 rounded-full bg-muted-foreground/30" />
            <span>IDLE</span>
          </div>
        </div>
      </div>
    );
  }

  const isClean = result.prediction === "clean";
  const { probability } = getVerdictView(result);
  const percentage = Math.round(probability * 100);

  const shap = isShapExplanation(result.explanation) ? result.explanation : null;
  const topTokens: ShapTopToken[] = shap
    ? shap.top_tokens.filter(isShapTopToken)
    : [];

  const explanationType =
    result.explanation &&
      typeof result.explanation === "object" &&
      typeof (result.explanation as Record<string, unknown>)["type"] === "string"
      ? ((result.explanation as Record<string, unknown>)["type"] as string)
      : null;

  const ruleJustification =
    result.rule_justification &&
      typeof result.rule_justification === "object" &&
      typeof (result.rule_justification as Record<string, unknown>)["name"] === "string" &&
      typeof (result.rule_justification as Record<string, unknown>)["reason"] === "string"
      ? (result.rule_justification as Record<string, unknown>)
      : null;

  // ML explanation fields from updated backend contract
  const mlSummary = typeof result.explanation_summary === "string" && result.explanation_summary.trim()
    ? result.explanation_summary.trim()
    : null;
  const mlTokens = Array.isArray(result.explanation_tokens) ? result.explanation_tokens : [];

  // Justification dispatcher precedence:
  // 1) ML explanation summary (new contract)
  // 2) Model `explanation` (legacy SHAP object)
  // 3) `rule_justification`
  // 4) Nothing
  const justificationKind: "ml_explanation" | "model_shap" | "model_other" | "rule" | null = mlSummary
    ? "ml_explanation"
    : shap
      ? "model_shap"
      : explanationType
        ? "model_other"
        : ruleJustification
          ? "rule"
          : null;

  return (
    <div
      className={cn(
        "relative rounded-2xl p-8 overflow-hidden animate-scale-in transition-all duration-500 min-h-[340px]",
        isClean
          ? "glass border-2 border-success/50 glow-success"
          : "glass border-2 border-destructive/50 glow-destructive"
      )}
    >
      {/* Background gradient */}
      <div
        className={cn(
          "absolute inset-0 opacity-20",
          isClean
            ? "bg-gradient-to-br from-success/30 via-transparent to-success/10"
            : "bg-gradient-to-br from-destructive/30 via-transparent to-destructive/10"
        )}
      />

      {/* Corner decorations */}
      <div className={cn("absolute top-3 left-3 w-8 h-8 border-t-2 border-l-2", isClean ? "border-success/60" : "border-destructive/60")} />
      <div className={cn("absolute top-3 right-3 w-8 h-8 border-t-2 border-r-2", isClean ? "border-success/60" : "border-destructive/60")} />
      <div className={cn("absolute bottom-3 left-3 w-8 h-8 border-b-2 border-l-2", isClean ? "border-success/60" : "border-destructive/60")} />
      <div className={cn("absolute bottom-3 right-3 w-8 h-8 border-b-2 border-r-2", isClean ? "border-success/60" : "border-destructive/60")} />

      <div className="relative flex flex-col items-center justify-center gap-5">
        {/* Status badge */}
        <div className={cn(
          "px-3 py-1 rounded-full text-xs font-mono uppercase tracking-wider",
          isClean
            ? "bg-success/20 text-success border border-success/30"
            : "bg-destructive/20 text-destructive border border-destructive/30"
        )}>
          {isClean ? "SECURE" : "VULNERABLE"}
        </div>

        {/* Icon */}
        <div className={cn(
          "relative p-5 rounded-2xl",
          isClean ? "bg-success/10" : "bg-destructive/10"
        )}>
          <div className={cn(
            "absolute inset-0 rounded-2xl pulse-glow",
            isClean ? "bg-success/20" : "bg-destructive/20"
          )} />
          {isClean ? (
            <ShieldCheck className="relative w-16 h-16 text-success" />
          ) : (
            <ShieldAlert className="relative w-16 h-16 text-destructive" />
          )}
        </div>

        {/* Result text */}
        <div className="text-center space-y-1">
          <p className={cn(
            "text-3xl font-bold uppercase tracking-wide text-glow",
            isClean ? "text-success" : "text-destructive"
          )}>
            {isClean ? "Clean Code" : "Defect Found"}
          </p>
          <p className="text-sm text-muted-foreground">
            {isClean
              ? "No security vulnerabilities detected"
              : "Potential security vulnerability identified"}
          </p>
        </div>

        {/* Confidence meter */}
        <div className="w-full max-w-xs space-y-3 mt-2">
          <div className="flex justify-between items-center text-sm">
            <span className="text-muted-foreground font-medium inline-flex items-center gap-1.5">
              Confidence Score
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    aria-label="Confidence details"
                  >
                    <Info className="w-3.5 h-3.5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Backend-reported confidence value for the returned prediction (rule-based overrides may force this).
                </TooltipContent>
              </Tooltip>
            </span>
            <span className={cn(
              "font-mono font-bold text-lg",
              isClean ? "text-success" : "text-destructive"
            )}>
              {percentage}%
            </span>
          </div>
          <div className="relative h-3 bg-muted/30 rounded-full overflow-hidden">
            <div
              className={cn(
                "absolute inset-y-0 left-0 rounded-full transition-all duration-1000 ease-out",
                isClean
                  ? "bg-gradient-to-r from-success/80 to-success"
                  : "bg-gradient-to-r from-destructive/80 to-destructive"
              )}
              style={{ width: `${percentage}%` }}
            />
            {/* Shine effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent shimmer" />
          </div>
        </div>

        {/* Backend authority + explainability (additive) */}
        <div className="w-full max-w-xs mt-5 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div className="inline-flex items-center gap-1.5 text-sm text-muted-foreground font-medium">
              Decision Source
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                    aria-label="Decision source details"
                  >
                    <Info className="w-3.5 h-3.5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  Provided verbatim by the backend to indicate which authority produced the final decision.
                </TooltipContent>
              </Tooltip>
            </div>
            <Badge variant="secondary" className="font-mono text-[10px] max-w-[220px] truncate">
              {result.decision_source}
            </Badge>
          </div>

          {justificationKind === "ml_explanation" ? (
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="ml-explanation" className="border-border/40">
                <AccordionTrigger className="py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span>ML Explanation</span>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      neural-analysis
                    </Badge>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2">
                  <div className="space-y-3">
                    {/* Summary */}
                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground inline-flex items-center gap-1.5">
                        Analysis Summary
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                              aria-label="Summary details"
                            >
                              <Info className="w-3.5 h-3.5" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            Backend-generated explanation of why this code may contain a defect.
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <p className="text-xs text-foreground/90 leading-snug">{mlSummary}</p>
                    </div>

                    {/* Tokens */}
                    {mlTokens.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span className="inline-flex items-center gap-1.5">
                            Top tokens (impact)
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  type="button"
                                  className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                                  aria-label="Token impact details"
                                >
                                  <Info className="w-3.5 h-3.5" />
                                </button>
                              </TooltipTrigger>
                              <TooltipContent>
                                Signed influence values (normalized); not probabilities.
                              </TooltipContent>
                            </Tooltip>
                          </span>
                          <span className="font-mono">{mlTokens.length}</span>
                        </div>
                        <div className="space-y-2">
                          {(() => {
                            const maxAbs = Math.max(...mlTokens.map((t) => Math.abs(t.impact)), 0);
                            return mlTokens.map((t, idx) => {
                              const widthPct = maxAbs > 0 ? (Math.abs(t.impact) / maxAbs) * 100 : 0;
                              const sign = t.impact >= 0 ? "+" : "";
                              return (
                                <div key={`${t.token}-${idx}`} className="space-y-1">
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="font-mono text-xs truncate">{t.token}</span>
                                    <span className="font-mono text-xs text-muted-foreground">
                                      {sign}{t.impact.toFixed(4)}
                                    </span>
                                  </div>
                                  <div className="h-1.5 bg-muted/30 rounded-full overflow-hidden">
                                    <div className="h-full bg-primary/70" style={{ width: `${widthPct}%` }} />
                                  </div>
                                </div>
                              );
                            });
                          })()}
                        </div>
                      </div>
                    )}
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          ) : justificationKind === "model_shap" ? (
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="model-explanation" className="border-border/40">
                <AccordionTrigger className="py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span>Model Inference</span>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {shap?.type}
                    </Badge>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1.5">
                        Inference confidence
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                              aria-label="Model confidence details"
                            >
                              <Info className="w-3.5 h-3.5" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>
                            Backend-provided confidence for the model inference (separate from the displayed verdict confidence).
                          </TooltipContent>
                        </Tooltip>
                      </span>
                      <span className="font-mono">{Math.round((shap?.model_confidence ?? 0) * 100)}%</span>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span className="inline-flex items-center gap-1.5">
                          Top tokens (impact)
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                type="button"
                                className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                                aria-label="Impact details"
                              >
                                <Info className="w-3.5 h-3.5" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>
                              Signed influence values (normalized); not probabilities.
                            </TooltipContent>
                          </Tooltip>
                        </span>
                        <span className="font-mono">{topTokens.length}</span>
                      </div>

                      {topTokens.length === 0 ? (
                        <div className="text-xs text-muted-foreground/80">
                          No token attributions available for this input.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {(() => {
                            const maxAbs = Math.max(...topTokens.map((t) => Math.abs(t.impact)), 0);
                            return topTokens.map((t, idx) => {
                              const widthPct = maxAbs > 0 ? (Math.abs(t.impact) / maxAbs) * 100 : 0;
                              const sign = t.impact >= 0 ? "+" : "";
                              return (
                                <div key={`${t.token}-${idx}`} className="space-y-1">
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="font-mono text-xs truncate">{t.token}</span>
                                    <span className="font-mono text-xs text-muted-foreground">
                                      {sign}
                                      {t.impact.toFixed(4)}
                                    </span>
                                  </div>
                                  <div className="h-1.5 bg-muted/30 rounded-full overflow-hidden">
                                    <div className="h-full bg-primary/70" style={{ width: `${widthPct}%` }} />
                                  </div>
                                </div>
                              );
                            });
                          })()}
                        </div>
                      )}
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          ) : justificationKind === "model_other" ? (
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-muted-foreground">Model Inference</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Badge variant="outline" className="font-mono text-[10px] cursor-help max-w-[220px] truncate">
                    {explanationType}
                  </Badge>
                </TooltipTrigger>
                <TooltipContent>Backend attached an unsupported explanation type.</TooltipContent>
              </Tooltip>
            </div>
          ) : justificationKind === "rule" ? (
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="rule-justification" className="border-border/40">
                <AccordionTrigger className="py-2 text-sm">
                  <div className="flex items-center gap-2">
                    <span>Triggered Rule</span>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      rule-based
                    </Badge>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                      <span className="inline-flex items-center gap-1.5">
                        Trigger details
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              className="inline-flex items-center justify-center rounded p-0.5 text-muted-foreground/70 hover:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                              aria-label="Rule attribute details"
                            >
                              <Info className="w-3.5 h-3.5" />
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>Provided verbatim by the backend for rule-based decisions.</TooltipContent>
                        </Tooltip>
                      </span>
                      <div className="flex items-center gap-2">
                        {typeof ruleJustification?.["verdict"] === "string" ? (
                          <Badge variant="secondary" className="font-mono text-[10px]">
                            {String(ruleJustification["verdict"])}
                          </Badge>
                        ) : null}
                        {typeof ruleJustification?.["rule_type"] === "string" ? (
                          <Badge variant="secondary" className="font-mono text-[10px]">
                            {String(ruleJustification["rule_type"])}
                          </Badge>
                        ) : null}
                        {typeof ruleJustification?.["certainty"] === "string" ? (
                          <Badge variant="secondary" className="font-mono text-[10px]">
                            {String(ruleJustification["certainty"]) === "certain"
                              ? "GUARANTEED"
                              : String(ruleJustification["certainty"]).toUpperCase()}
                          </Badge>
                        ) : typeof ruleJustification?.["confidence"] === "string" ? (
                          <Badge variant="secondary" className="font-mono text-[10px]">
                            {String(ruleJustification["confidence"])}
                          </Badge>
                        ) : null}
                      </div>
                    </div>

                    {typeof ruleJustification?.["category"] === "string" ? (
                      <div className="space-y-1">
                        <div className="text-xs text-muted-foreground">Category</div>
                        <div className="font-mono text-xs truncate">{String(ruleJustification["category"])}</div>
                      </div>
                    ) : null}

                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">Triggered rule</div>
                      <div className="font-mono text-xs truncate">{String(ruleJustification?.["name"])}</div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-xs text-muted-foreground">Why it triggered</div>
                      <div className="text-xs text-foreground/90 leading-snug">{String(ruleJustification?.["reason"])}</div>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
