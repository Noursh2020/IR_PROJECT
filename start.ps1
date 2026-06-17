# start.ps1 — تشغيل كل خدمات IR System دفعة واحدة
# الاستخدام: .\start.ps1

Write-Host "Starting IR System Services..." -ForegroundColor Cyan

$services = @(
    @{ name = "Preprocessing  "; port = 8001; module = "services.preprocessing.main:app" },
    @{ name = "Indexing       "; port = 8002; module = "services.indexing.main:app" },
    @{ name = "Embedding      "; port = 8006; module = "services.embedding.main:app" },
    @{ name = "Retrieval      "; port = 8003; module = "services.retrieval.main:app" },
    @{ name = "Ranking & Eval "; port = 8004; module = "services.ranking_evaluation.main:app" },
    @{ name = "Query Refinement"; port = 8005; module = "services.query_refinement.main:app" },
    @{ name = "Document Store "; port = 8009; module = "services.document_store.main:app" },
    @{ name = "Dataset Loader "; port = 8007; module = "services.dataset_loader.main:app" },
    @{ name = "RAG            "; port = 8008; module = "services.rag.main:app" },
    @{ name = "Gateway        "; port = 8000; module = "services.gateway.main:app" }
)

foreach ($svc in $services) {
    Start-Process powershell -ArgumentList `
        "-NoExit", "-Command", `
        "Write-Host '$($svc.name) :$($svc.port)' -ForegroundColor Green; uvicorn $($svc.module) --port $($svc.port) --reload" `
        -WindowStyle Normal
    Start-Sleep -Milliseconds 800
}

Write-Host ""
Write-Host "All services started!" -ForegroundColor Green
Write-Host "Gateway: http://localhost:8000" -ForegroundColor Yellow
Write-Host "UI:      open ui\index.html in browser" -ForegroundColor Yellow
Write-Host ""
Write-Host "To check health: Invoke-RestMethod http://localhost:8000/health" -ForegroundColor Cyan