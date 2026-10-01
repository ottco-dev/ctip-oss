# Security policy

CTIP is meant to run on a lab workstation or inside a trusted network. The API can start containers, read and write
datasets and run models — **do not expose it to the internet without authentication and TLS** (see
`docs/deployment/network_setup.md`).

## Reporting a vulnerability

Please **do not open a public issue**. Report privately through GitHub:
**Security → Report a vulnerability** on <https://github.com/ottco-dev/ctip-oss/security/advisories/new>.

Include what is affected, how to reproduce it and the impact you expect. You will get an answer within 7 days.

## Supported versions

Only the latest `main` receives fixes while CTIP is in alpha.
