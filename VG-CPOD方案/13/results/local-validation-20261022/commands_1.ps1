$cpfServer = "xhw@211.87.232.122"
$cpfZip = "C:\Users\llenovo\Documents\ChatGPT\USG\output\trustcpf_conditional_v2.zip"

ssh $cpfServer 'mkdir -p "$HOME/experiments"'
scp "$cpfZip" "${cpfServer}:~/experiments/trustcpf_conditional_v2.zip"
if ($LASTEXITCODE -ne 0) { throw "上传未完成，请重新执行 scp" }

ssh $cpfServer