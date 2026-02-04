import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { 
  Play, 
  Trash2, 
  Github, 
  Sparkles, 
  Bug, 
  CheckCircle2,
  Cpu,
  Shield,
  Zap,
  ChevronRight
} from "lucide-react";
import CodeEditor from "@/components/CodeEditor";
import PredictionResult from "@/components/PredictionResult";
import AnimatedBackground from "@/components/AnimatedBackground";
import { PredictResponse, getVerdictView, predict } from "@/lib/mockPredict";

// Backend URL is defined in `src/lib/mockPredict.ts`.

const SAMPLE_CODE = `// Example: Buffer overflow vulnerability
void vulnerable_function(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // No bounds checking!
    printf("Buffer: %s\\n", buffer);
}

int main() {
    char *user_input = "This is safe input";
    vulnerable_function(user_input);
    return 0;
}`;

const CLEAN_CODE = `// Example: Safe swap function
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 5, y = 10;
    swap(&x, &y);
    return 0;
}`;

const Index = () => {
  const [code, setCode] = useState(SAMPLE_CODE);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // 🔽 REPLACED mockPredict WITH REAL BACKEND CALL
  const analyzeWithBackend = async (code: string): Promise<PredictResponse> => {
    return predict(code);
  };

  const handleAnalyze = async () => {
    if (!code.trim()) {
      toast({
        title: "No code provided",
        description: "Please enter some C/C++ code to analyze.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setResult(null);

    try {
      const prediction = await analyzeWithBackend(code);
      setResult(prediction);

      const { label, probability } = getVerdictView(prediction);

      toast({
        title:
          label === "clean"
            ? "✓ Code is Clean"
            : "⚠ Defect Detected",
        description: `Confidence: ${Math.round(
          probability * 100
        )}%`,
      });
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: "Could not connect to backend. Is the server running?",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    setCode("");
    setResult(null);
  };

  const loadSample = (type: "vulnerable" | "clean") => {
    setCode(type === "vulnerable" ? SAMPLE_CODE : CLEAN_CODE);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <AnimatedBackground />
      <div className="fixed inset-0 gradient-mesh pointer-events-none z-0" />
      <div className="fixed inset-0 grid-bg opacity-20 pointer-events-none z-0" />

      {/* Header */}
      <header className="relative z-10 border-b border-border/50 glass-strong sticky top-0">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4 animate-slide-up">
            <div className="relative p-2.5 rounded-xl bg-primary/10 border border-primary/30 glow-primary">
              <Shield className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground flex items-center gap-2">
                DefectPrediction
                <span className="px-2 py-0.5 text-[10px] font-mono uppercase bg-primary/20 text-primary rounded-full border border-primary/30">
                  AI
                </span>
              </h1>
              <p className="text-xs text-muted-foreground">
                GraphCodeBERT Neural Vulnerability Scanner
              </p>
            </div>
          </div>
          <a
            href="https://github.com/SreehariU/DefectPrediction"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-all duration-300 group"
          >
            <Github className="w-5 h-5 group-hover:scale-110 transition-transform" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-4 py-10">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-5">
              <CodeEditor value={code} onChange={setCode} disabled={isLoading} />
              <div className="flex gap-3">
                <Button
                  onClick={handleAnalyze}
                  disabled={isLoading || !code.trim()}
                  className="flex-1 h-14"
                >
                  <Zap className="w-5 h-5 mr-2" />
                  {isLoading ? "Analyzing..." : "Analyze Code"}
                </Button>
                <Button variant="outline" onClick={handleClear}>
                  <Trash2 />
                </Button>
              </div>
            </div>

            <PredictionResult result={result} isLoading={isLoading} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
