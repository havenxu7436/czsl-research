param(
    [ValidateSet('check','create','verify')][string]$Mode = 'check',
    [string]$Repository = 'czsl-research'
)
$ErrorActionPreference = 'Stop'
$expectedOwner = 'havenxu7436'
$env:GCM_INTERACTIVE = 'Never'
$env:GIT_TERMINAL_PROMPT = '0'
# Retrieve the user's existing Git credential in memory; never print or save it.
$credentialLines = @('protocol=https','host=github.com',"username=$expectedOwner",'') | git -c credential.interactive=never credential fill 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Existing GitHub sign-in is unavailable.' }
$credentialFields = @{}
foreach ($credentialLine in $credentialLines) {
    $pair = $credentialLine -split '=',2
    if ($pair.Count -eq 2) { $credentialFields[$pair[0]] = $pair[1] }
}
if (-not $credentialFields['password']) { throw 'Existing GitHub credential is empty.' }
$apiHeaders = @{ Authorization='Bearer ' + $credentialFields['password']; Accept='application/vnd.github+json'; 'X-GitHub-Api-Version'='2022-11-28'; 'User-Agent'='CZSL-private-archive' }
$user = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $apiHeaders
if ($user.login -ne $expectedOwner -or $user.type -ne 'User') { throw 'GitHub account does not match the personal owner.' }
if ($Mode -eq 'check') { @{owner=$user.login;personal_account=$true;authenticated=$true} | ConvertTo-Json; exit }
if ($Repository -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'Invalid repository name.' }
$repoUrl = "https://api.github.com/repos/$expectedOwner/$Repository"
if ($Mode -eq 'create') {
    $existing = $null
    try { $existing=Invoke-RestMethod -Uri $repoUrl -Headers $apiHeaders } catch {
        if ([int]$_.Exception.Response.StatusCode -ne 404) { throw 'Could not safely check the target repository.' }
    }
    if ($existing) { throw 'Target repository already exists; preserve it and select another name.' }
    $body = @{name=$Repository;description='Private CZSL research archive: chronological experiments, residual and formula-aligned routes, evidence and audits';private=$true;auto_init=$false;has_issues=$true;has_wiki=$false} | ConvertTo-Json
    $repo = Invoke-RestMethod -Uri 'https://api.github.com/user/repos' -Headers $apiHeaders -Method Post -Body $body -ContentType 'application/json'
} else { $repo=Invoke-RestMethod -Uri $repoUrl -Headers $apiHeaders }
if (-not $repo.private -or $repo.owner.login -ne $expectedOwner -or $repo.visibility -ne 'private') { throw 'The repository is not the expected personal PRIVATE repository.' }
$out = [ordered]@{owner=$repo.owner.login;name=$repo.name;private=$repo.private;visibility=$repo.visibility;html_url=$repo.html_url;clone_url=$repo.clone_url;default_branch=$repo.default_branch}
if ($Mode -eq 'verify') {
    $ref=Invoke-RestMethod -Uri "$repoUrl/git/ref/heads/main" -Headers $apiHeaders
    $out.remote_main_sha=$ref.object.sha
    $out.local_main_sha=(git rev-parse main).Trim()
    if ($out.remote_main_sha -ne $out.local_main_sha) { throw 'Local and GitHub main commits differ.' }
}
$out | ConvertTo-Json
