#!/bin/sh
set -eu

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"

exec "$@"
