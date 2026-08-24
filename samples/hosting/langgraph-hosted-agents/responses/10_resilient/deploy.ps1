[CmdletBinding()]
param(
    [string] $Environment = "resilient",
    [string] $SubscriptionId,
    [string] $Location
)

$ErrorActionPreference = "Stop"

$sampleRoot = $PSScriptRoot

if (-not (Get-Command "azd" -ErrorAction SilentlyContinue)) {
    throw "Required command 'azd' was not found on PATH."
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

if ($SubscriptionId) {
    & azd env set AZURE_SUBSCRIPTION_ID $SubscriptionId
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set AZURE_SUBSCRIPTION_ID for environment '$Environment'."
    }
}
if ($Location) {
    & azd env set AZURE_LOCATION $Location
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set AZURE_LOCATION for environment '$Environment'."
    }
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