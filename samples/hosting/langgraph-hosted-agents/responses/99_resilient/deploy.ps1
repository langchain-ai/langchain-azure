[CmdletBinding()]
param(
    [string] $Environment = "ai-test",
    [string] $ProjectEndpoint,
    [string] $ProjectId,
    [string] $SubscriptionId,
    [string] $TenantId,
    [string] $Location
)

$ErrorActionPreference = "Stop"

$serviceName = "langchain-azure-resilient-responses"
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
    & azd @newEnvironmentArgs
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize azd environment '$Environment'."
}

$environmentValues = @{
    AZURE_AI_PROJECT_ENDPOINT = $ProjectEndpoint
    FOUNDRY_PROJECT_ENDPOINT = $ProjectEndpoint
    AZURE_AI_PROJECT_ID = $ProjectId
    AZURE_SUBSCRIPTION_ID = $SubscriptionId
    AZURE_TENANT_ID = $TenantId
    AZURE_LOCATION = $Location
}
foreach ($entry in $environmentValues.GetEnumerator()) {
    if ($entry.Value) {
        & azd env set $entry.Key $entry.Value
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to set azd environment value '$($entry.Key)'."
        }
    }
}

$savedProjectEndpoint = & azd env get-value FOUNDRY_PROJECT_ENDPOINT 2>$null
if ($LASTEXITCODE -ne 0 -or -not $savedProjectEndpoint) {
    throw "No Foundry project is configured. On the first run pass -ProjectEndpoint, -ProjectId, -SubscriptionId, -TenantId, and -Location."
}

& azd deploy $serviceName --no-prompt
if ($LASTEXITCODE -ne 0) {
    throw "Deployment failed."
}