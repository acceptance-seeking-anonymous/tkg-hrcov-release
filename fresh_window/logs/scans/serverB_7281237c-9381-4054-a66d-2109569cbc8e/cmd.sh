set +e
if command -v docker >/dev/null 2>&1 && docker image inspect tkg-gpu:latest >/dev/null 2>&1; then
  docker run --rm -v /home/[REDACTED:~ANON]:/home/[REDACTED:~ANON]:ro -v $PWD:/workspace -w /workspace tkg-gpu:latest python3 fw_uninspected_scan.py serverB /home/[REDACTED:~ANON] > scan.json 2> scan.err
  echo "mode=docker-root-readonly" > scan_mode.txt
else
  python3 fw_uninspected_scan.py serverB /home/[REDACTED:~ANON] > scan.json 2> scan.err
  echo "mode=host-user" > scan_mode.txt
fi
ls -la; wc -c scan.json
