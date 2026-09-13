#!/usr/bin/env bash
# Pull the RHUH-GBM NIfTI package (40 preoperative + postoperative + follow-up
# GBM studies, 2.9 GB, CC BY 4.0) from TCIA.
#
# TCIA hands this one out through Aspera Faspex 5, not plain HTTP. The web page
# says "IBM Aspera Connect plugin required", and the server confirms it:
# /api/v5/configuration reports "http_gateway_url": null, so there is no HTTP
# fallback -- the bytes only move over FASP (UDP 33001). What IS scriptable is
# everything up to the transfer: the public link carries a passcode, which the
# OAuth public-link flow trades for a bearer token, which yields a transfer
# spec with a one-shot ATM2 token. This script does that part and then hands
# the spec to ascp.
#
# Needs ascp on PATH (or ASCP set). Either install IBM Aspera Connect, or
# unpack the Aspera transfer SDK and point ASCP at its ascp.exe.
#
# It also needs an SSH private key, which is the non-obvious part. The FASP
# server offers exactly one auth method -- `publickey` -- so the ATM2 token
# cannot authenticate the SSH layer no matter how it is passed (ASPERA_SCP_PASS
# and -W both fail with "No methods left to try"). The two credentials do
# different jobs: the well-known Aspera token-auth key gets you through SSH,
# then the token in ASPERA_SCP_TOKEN authorises this specific package. That key
# is deliberately public -- it is shipped inside every Aspera Connect install
# and inside the aspera-cli gem -- but it is NOT in the transfer SDK zip, which
# is why an SDK-only setup fails at the SSH handshake.
#
#   bash scripts/fetch_rhuh.sh                 # -> data/external/rhuh_raw
#   ASCP=/c/path/to/ascp.exe bash scripts/fetch_rhuh.sh
set -euo pipefail

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
DEST="${DEST:-$DRIVE/data/external/rhuh_raw}"
# python here is the Windows interpreter, which cannot see Git Bash's /tmp
TS="${TMPDIR:-${TMPDIR:-/tmp}}/rhuh_ts.json"
CONNECT='/c/Program Files/IBM/Aspera Connect'
ASCP="${ASCP:-$(command -v ascp || true)}"
[ -n "$ASCP" ] || ASCP="$CONNECT/bin/ascp.exe"
# Our own copy first so the script survives Connect being uninstalled; the
# Connect-installed key is the same credential under its official name.
KEY="${ASPERA_KEY:-$DRIVE/tools/aspera/aspera_bypass_rsa.pem}"
[ -f "$KEY" ] || KEY="$CONNECT/etc/aspera_tokenauth_id_rsa"
HOST=https://faspex.cancerimagingarchive.net
FX=$HOST/aspera/faspex
# The public link carries a client id and a base64 context blob holding the
# package passcode. Both belong to the TCIA download link rather than to this
# project, and TCIA rotates them, so they are not stored here.
#
#   1. open the RHUH-GBM page on TCIA and follow its "Download" link
#   2. the browser lands on a faspex URL of the form
#        .../auth/authorize_public_link?...&client_id=<CID>&state=<CTX>
#   3. copy those two values into the environment:
#        export RHUH_CLIENT_ID=<CID> RHUH_CONTEXT=<CTX>
CID="${RHUH_CLIENT_ID:-}"
CTX="${RHUH_CONTEXT:-}"
[ -n "$CID" ] && [ -n "$CTX" ] || {
  echo "ERROR: set RHUH_CLIENT_ID and RHUH_CONTEXT from the TCIA public link;" \
       "see the comment at the top of this script" >&2; exit 1; }
PKG=684

say() { echo "[$(date '+%H:%M:%S')] [rhuh] $*"; }

[ -x "$ASCP" ] || { say "ERROR: ascp not found. Install Aspera Connect or set ASCP=/path/to/ascp.exe"; exit 1; }
[ -f "$KEY" ] || { say "ERROR: no Aspera token-auth key. Install Aspera Connect, or set ASPERA_KEY=/path/to/key"; exit 1; }

say "authorising against the public link"
LOC=$(curl -sS -o /dev/null -w '%{redirect_url}' \
  "$FX/auth/authorize_public_link?response_type=code&client_id=$CID&redirect_uri=%2Faspera%2Ffaspex%2Ftoken&state=$CTX")
CODE=$(printf '%s' "$LOC" | sed -n 's/.*code=\([^&]*\).*/\1/p')
[ -n "$CODE" ] || { say "ERROR: no authorization code (public link may have been rotated)"; exit 1; }

TOK=$(curl -sS -X POST "$FX/auth/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data "grant_type=authorization_code&client_id=$CID&redirect_uri=%2Faspera%2Ffaspex%2Ftoken&code=$CODE&state=$CTX" \
  | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p')
[ -n "$TOK" ] || { say "ERROR: token exchange failed"; exit 1; }

say "fetching the transfer spec for package $PKG"
curl -sS -X POST -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' \
  "$FX/api/v5/packages/$PKG/transfer_spec/download?transfer_type=connect&type=received" -d '{}' > "$TS"

mkdir -p "$DEST"
eval "$(TS_WIN="$(cygpath -w "$TS")" python - <<'PY'
import json, os, shlex
ts = json.load(open(os.environ["TS_WIN"]))
if 'token' not in ts:
    raise SystemExit("echo 'ERROR: transfer spec has no token'; exit 1")
print(f"REMOTE_HOST={shlex.quote(ts['remote_host'])}")
print(f"REMOTE_USER={shlex.quote(ts['remote_user'])}")
print(f"FASP_PORT={ts.get('fasp_port', 33001)}")
print(f"SSH_PORT={ts.get('ssh_port', 33001)}")
print(f"ATM_TOKEN={shlex.quote(ts['token'])}")
print(f"SRC={shlex.quote(ts['paths'][0]['source'])}")
print(f"COOKIE={shlex.quote(ts.get('cookie',''))}")
PY
)"

# ascp.exe is a native Windows binary; it cannot read a Git Bash path
DEST_WIN="$(cygpath -w "$DEST")"
KEY_WIN="$(cygpath -w "$KEY")"
say "ascp -> $DEST_WIN  (2.9 GB / 720 files; needs outbound UDP $FASP_PORT)"
export ASPERA_SCP_TOKEN="$ATM_TOKEN" ASPERA_SCP_COOKIE="$COOKIE"
# $SRC is a path on the FASP server, but it starts with "/" and MSYS rewrites
# any such argument into a Windows path before the native exe ever sees it --
# ascp would look for "C:/Program Files/RHUH-...". MSYS_NO_PATHCONV stops that.
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
"$ASCP" -T --policy=fair -l 100M -m 1M -P "$SSH_PORT" -O "$FASP_PORT" \
        --mode=recv --user="$REMOTE_USER" --host="$REMOTE_HOST" \
        -k 2 --overwrite=diff -i "$KEY_WIN" \
        "$SRC" "$DEST_WIN"

say "done. $(find "$DEST" -name '*.nii.gz' | wc -l) NIfTI files under $DEST"
say "next: python scripts/prep_rhuh.py"
