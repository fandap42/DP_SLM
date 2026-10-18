param(
    [Parameter(Mandatory=$true)][string]$InputPptx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)

$InputPptx = [System.IO.Path]::GetFullPath($InputPptx)
$OutputPdf = [System.IO.Path]::GetFullPath($OutputPdf)

$ppt = New-Object -ComObject PowerPoint.Application
try {
    # Open presentation hidden / read-only
    $pres = $ppt.Presentations.Open($InputPptx, 1, 0, 0)
    # 32 = ppSaveAsPDF
    $pres.SaveAs($OutputPdf, 32)
    $pres.Close()
    Write-Host "PDF successfully generated: $OutputPdf"
} catch {
    Write-Error "Failed to export PDF: $_"
    exit 1
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}
