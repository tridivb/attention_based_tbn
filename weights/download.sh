# wget and zip required
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zdo43oc2wiUgNMNiPc45OpECBHHwQgd4' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zdo43oc2wiUgNMNiPc45OpECBHHwQgd4" -O weights.zip
rm -rf /tmp/cookies.txt
unzip weights.zip