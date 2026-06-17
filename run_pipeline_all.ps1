$CondaBase = (conda info --base).Trim()
$PYTHON = Join-Path $CondaBase "envs/scanvi311/python.exe"
$LOG    = "logs/pipeline_all_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Assert-Success {
    # Tee-Object 不是原生可执行文件，不会重置 $LASTEXITCODE，因此这里读到的仍是
    # 管道里前一个 python.exe 进程的真实退出码。任意一步失败立即停止整个批处理。
    param($Description)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[FATAL] $Description 失败，exit code=$LASTEXITCODE，立即停止。" | Tee-Object -Append -FilePath $LOG
        exit $LASTEXITCODE
    }
}

function Run-Pipeline {
    param($Config, $Seed, $Rts)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Config  seed=$Seed  rts=$Rts"
    & $PYTHON run_pipeline.py --config $Config --seed $Seed --rare_train_size $Rts 2>&1 |
        Tee-Object -Append -FilePath $LOG
    Assert-Success "run_pipeline.py $Config seed=$Seed rts=$Rts"
}

Write-Host "=== Pipeline for all 5 datasets (生成 embeddings) ===" | Tee-Object -FilePath $LOG
Write-Host "Log: $LOG"

foreach ($cfg in @(
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_lung_stroma.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/tabula_sapiens_stomach.yaml"
)) {
    foreach ($seed in 42,43,44) {
        foreach ($rts in "0.01","0.05","0.10","all") {
            Run-Pipeline $cfg $seed $rts
        }
    }
}

Write-Host "`n=== 所有 Pipeline 完成，开始跑解耦后的 5 个对比方法脚本 ===" | Tee-Object -Append -FilePath $LOG
foreach ($script in @(
    "tools/comparison/run_scanvi_comparison.py",
    "tools/comparison/run_knn_comparison.py",
    "tools/comparison/run_celltypist_comparison.py",
    "tools/comparison/run_scbalance_comparison.py",
    "tools/comparison/run_scrarerefine_comparison.py"
)) {
    Write-Host "`n--- $script ---" | Tee-Object -Append -FilePath $LOG
    & $PYTHON $script 2>&1 | Tee-Object -Append -FilePath $LOG
    Assert-Success $script
}

Write-Host "`n=== 5 个对比方法脚本完成，跑 scCAD ===" | Tee-Object -Append -FilePath $LOG
& $PYTHON tools/comparison/run_scCAD_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG
Assert-Success "tools/comparison/run_scCAD_comparison.py"

$SB = Join-Path $CondaBase "envs/sandbox310/python.exe"
Write-Host "`n=== 跑 ProtoCloud (sandbox310) ===" | Tee-Object -Append -FilePath $LOG
& $SB tools/comparison/run_protocloud_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG
Assert-Success "tools/comparison/run_protocloud_comparison.py"

Write-Host "`n=== 跑 HiCat (sandbox310, transductive) ===" | Tee-Object -Append -FilePath $LOG
& $SB tools/comparison/run_hicat_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG
Assert-Success "tools/comparison/run_hicat_comparison.py"

Write-Host "`n=== 跑 TOSICA (sandbox310) ===" | Tee-Object -Append -FilePath $LOG
& $SB tools/comparison/run_tosica_comparison.py 2>&1 | Tee-Object -Append -FilePath $LOG
Assert-Success "tools/comparison/run_tosica_comparison.py"

Write-Host "`n=== 全部完成 ===" | Tee-Object -Append -FilePath $LOG
