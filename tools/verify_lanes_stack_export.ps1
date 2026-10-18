param([Parameter(Mandatory=$true)][string]$OutputDirectory,
      [string]$IndexSuffix='final')
# Export every first-parent change, including merges, against the full base.
# Preserve files deliberately absent from the slim source export.
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$original='4d1079249ad0cc2553a9632ac9ef267553913dd1'
$slim='base/2.8.7-slim'
$branch='lane/pool-verify-lanes-stack'
if($IndexSuffix -notmatch '^[a-z0-9-]+$'){throw 'Index suffix must be a plain task label'}
$taskIndex=Join-Path (Split-Path -Parent $repo) ('export-index-'+$IndexSuffix)
if(Test-Path -LiteralPath $taskIndex){throw 'Export index already exists; use a fresh owned index path'}
$previousIndex=$env:GIT_INDEX_FILE
$env:GIT_INDEX_FILE=$taskIndex
$utf8=[Text.UTF8Encoding]::new($false)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$records=@()
try {
    git -C $repo read-tree $original
    if($LASTEXITCODE){throw 'Cannot read the full original base tree'}
    $previousTree=(git -C $repo write-tree).Trim()
    $commits=@(git -C $repo rev-list --reverse --first-parent "$slim..$branch")
    $number=0
    foreach($commit in $commits){
        $parent=(git -C $repo rev-parse "$commit^1").Trim()
        $raw=@(git -C $repo diff-tree --raw --no-commit-id -r $parent $commit)
        $entries=@()
        foreach($row in $raw){
            if($row -notmatch '^:[0-7]{6} ([0-7]{6}) [0-9a-f]{40} ([0-9a-f]{40}) [A-Z]\t(.*)$'){
                throw ('Unrecognized changed path: '+$row)
            }
            $mode=$Matches[1];$blob=$Matches[2];$path=$Matches[3]
            if($mode -eq '000000'){$mode='0'}
            $entries+=($mode+' '+$blob+"`t"+$path)
        }
        if($entries.Count){
            # Windows PowerShell pipes CRLF, which Git treats as part of
            # each path. Write an explicit LF stream to the native process.
            $start=[Diagnostics.ProcessStartInfo]::new()
            $start.FileName=(Get-Command git).Source
            $start.Arguments='update-index --index-info'
            $start.WorkingDirectory=$repo
            $start.UseShellExecute=$false
            $start.RedirectStandardInput=$true
            $process=[Diagnostics.Process]::new()
            $process.StartInfo=$start
            [void]$process.Start()
            $process.StandardInput.Write(($entries -join "`n")+"`n")
            $process.StandardInput.Close()
            $process.WaitForExit()
            if($process.ExitCode){throw 'Cannot update the owned export index'}
        }
        $tree=(git -C $repo write-tree).Trim()
        if($tree -eq $previousTree){continue}
        $number++
        $filename=('{0:D4}.patch' -f $number)
        $destination=Join-Path $OutputDirectory $filename
        $diffFile=Join-Path (Split-Path -Parent $repo) ('export-diff-{0:D4}.patch' -f $number)
        git -C $repo diff --binary --full-index --no-ext-diff --output=$diffFile $previousTree $tree
        if($LASTEXITCODE){throw 'Cannot generate the tree change'}
        $header=@(git -C $repo log -1 --format=email $commit)
        $mail=($header -join "`n")+"`n`n"+[IO.File]::ReadAllText($diffFile)
        [IO.File]::WriteAllText($destination,$mail,$utf8)
        $records+=@{patch=$filename;lane_commit=$commit;before_tree=$previousTree;after_tree=$tree}
        $previousTree=$tree
    }
    $fullEntries=@{}
    foreach($line in @(git -C $repo ls-tree -r $previousTree)){
        $parts=$line -split "`t",2
        $fullEntries[$parts[1]]=$parts[0]
    }
    $retained=0
    foreach($line in @(git -C $repo ls-tree -r $branch)){
        $parts=$line -split "`t",2
        if($fullEntries[$parts[1]] -ne $parts[0]){throw ('Exported source differs: '+$parts[1])}
        $retained++
    }
    $manifest=@{engine_base=$original;slim_base=(git -C $repo rev-parse $slim).Trim();
                lane_tip=(git -C $repo rev-parse $branch).Trim();full_source_tree=$previousTree;
                retained_files_checked=$retained;patches=$records;
                method='first-parent full-base overlays; merge changes included'}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'export.json'),
                           ($manifest|ConvertTo-Json -Depth 6)+"`n",$utf8)
    Write-Output ('Exported '+$number+' patches; full-source tree '+$previousTree)
} finally {
    $env:GIT_INDEX_FILE=$previousIndex
}
