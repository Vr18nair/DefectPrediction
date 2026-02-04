// GraphCodeBERT API integration
// Backend: http://127.0.0.1:8000 (FastAPI with your trained model)

export type BackendPrediction = "clean" | "defective";

export interface ShapTopToken {
  token: string;
  impact: number;
}

export interface ExplanationBase {
  type: string;
  [key: string]: unknown;
}

export interface ShapExplanation extends ExplanationBase {
  type: "shap";
  model_confidence: number;
  top_tokens: ShapTopToken[];
}

export type Explanation = ShapExplanation | ExplanationBase;

export interface RuleJustification {
  name: string;
  reason: string;
  [key: string]: unknown;
}

export interface PredictResponse {
  prediction: BackendPrediction;
  clean_probability: number;
  defect_probability: number;
  decision_source: string;
  explanation?: Explanation;
  rule_justification?: RuleJustification;
  // Allow future backend fields without breaking typing at call sites.
  [key: string]: unknown;
}

const API_URL = "http://127.0.0.1:8000";

export const predict = async (code: string): Promise<PredictResponse> => {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });

  if (!response.ok) {
    throw new Error("Failed to analyze code. Is the backend running?");
  }

  return response.json();
};

export function isMLDecisionSource(decisionSource: string): boolean {
  // Structural string check for currently-known `decision_source` values.
  // Do not infer policy/authority/explainability from this helper.
  return decisionSource === "ml_high_confidence" || decisionSource === "ml_low_confidence";
}

export function isShapExplanation(explanation: Explanation | undefined): explanation is ShapExplanation {
  // Structural type guard only; does not interpret meaning of fields.
  if (!explanation || explanation.type !== "shap") return false;
  const obj: Record<string, unknown> = explanation;
  return typeof obj["model_confidence"] === "number" && Array.isArray(obj["top_tokens"]);
}

export function isShapTopToken(value: unknown): value is ShapTopToken {
  // Structural type guard only; does not interpret `impact` semantics.
  if (!value || typeof value !== "object") return false;
  const obj = value as Record<string, unknown>;
  return typeof obj["token"] === "string" && typeof obj["impact"] === "number";
}

export function getVerdictView(result: Pick<PredictResponse, "prediction" | "clean_probability" | "defect_probability">): {
  label: BackendPrediction;
  probability: number;
} {
  // Pure selector: chooses the backend probability field that corresponds to `prediction`.
  if (result.prediction === "clean") {
    return { label: "clean", probability: result.clean_probability };
  }
  if (result.prediction === "defective") {
    return { label: "defective", probability: result.defect_probability };
  }
  throw new Error("Unexpected prediction value from backend");
}

// Backward-compatible export name (avoid breaking imports during transition).
export const mockPredict = predict;
