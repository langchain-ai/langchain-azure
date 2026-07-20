[CmdletBinding()]
param(
    [string] $Environment = "resilient",
    [string] $SubscriptionId,
    [string] $Location
)

$ErrorActionPreference = "Stop"

$sampleRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $sampleRoot "..\..\..\..\..")).Path
$packageRoot = Join-Path $repoRoot "libs\azure-ai"
$vendorRoot = Join-Path $sampleRoot "vendor"
$expectedWheel = Join-Path $vendorRoot "langchain_azure_ai-1.2.8-py3-none-any.whl"

foreach ($command in @("azd", "uv")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required command '$command' was not found on PATH."
    }
}

New-Item -ItemType Directory -Path $vendorRoot -Force | Out-Null
Remove-Item (Join-Path $vendorRoot "langchain_azure_ai-*.whl") -Force -ErrorAction SilentlyContinue

& uv build --wheel --out-dir $vendorRoot $packageRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build the local langchain-azure-ai wheel."
}
if (-not (Test-Path $expectedWheel)) {
    throw "Expected wheel '$expectedWheel' was not produced. Update deploy.ps1 and requirements.txt for the package version."
}

$environmentRoot = Join-Path $sampleRoot ".azure\$Environment"
if (Test-Path $environmentRoot) {
    & azd env select $Environment --no-prompt
} else {
    $newEnvironmentArgs = @("env", "new", $Environment, "--no-prompt")
    if ($SubscriptionId) {
        $newEnvironmentArgs += @("--subscription", $SubscriptionId)
    }
    if ($Location) {
        $newEnvironmentArgs += @("--location", $Location)
    }
    & azd @newEnvironmentArgs
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize azd environment '$Environment'."
}

Write-Host "Provisioning the model declared in azure.yaml..."
& azd provision --no-prompt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to provision the Foundry model deployment."
}

$serviceName = "langchain-azure-resilient-responses-steerable"
Write-Host "Deploying $serviceName..."
& azd deploy $serviceName --no-prompt
if ($LASTEXITCODE -ne 0) {
    throw "Deployment of '$serviceName' failed."
}