# User accounts (hosted instances)

A CTIP instance that several people use should run with **user accounts** instead of one shared API token.

```env
AUTH_MODE=accounts
ADMIN_USERNAME=alice          # first admin, created at start-up when no active admin exists
ADMIN_PASSWORD=<at least 10 characters>
API_TOKEN=<optional, for scripts and ctip-worker admin calls>
```

## Modes

| `AUTH_MODE` | Behaviour |
|---|---|
| `auto` (default) | `token` when `API_TOKEN` is set, otherwise `off` — the previous behaviour |
| `off` | No authentication (local development only) |
| `token` | One shared `API_TOKEN`; the web UI asks for it once and stores it in an HttpOnly cookie |
| `accounts` | Login with username and password; roles; the `API_TOKEN` keeps working for scripts with admin rights |

## Roles

| Role | Can |
|---|---|
| `member` | Use the platform: datasets, training, annotation, detection, reports, compute queue (read) |
| `admin` | Additionally: users and invitations, settings, setup wizard, containers, compute enrolment tokens, worker management, job and artifact writes |

The rules live in `backend/middleware/auth.py` (`ADMIN_RULES`) and are enforced on the server for every request;
the UI only hides what a member cannot use.

## Adding people

- **Users page → Create user:** CTIP generates a temporary password, shown once. The person must set their own
  password at first login.
- **Users page → Invitation link:** a single-use link (`/register?invite=…`, 1–30 days, role chosen by the admin).
  The person picks their own username and password.
- Admins can change roles, disable accounts and reset passwords. Disabling an account or resetting its password
  signs out all of its sessions. The last active admin cannot be demoted or disabled.

## Security design

- Passwords: scrypt (N=2¹⁵, r=8, p=1, 16-byte salt), at least 10 characters, not containing the username.
- Sessions: random 256-bit token in the `ctip_session` cookie (HttpOnly, SameSite=Strict, Secure behind HTTPS);
  only its SHA-256 is stored. 14 days sliding expiry; logout deletes the session server-side.
- Brute force: 5 failed logins per username or per IP address lock that key for 15 minutes. All failures return the
  same message and take the same time, so usernames cannot be probed.
- Invitations are stored as hashes, expire and work once.
- WebSockets check the session cookie in the handshake.
- `ADMIN_PASSWORD` is only used to create the first admin; it never overwrites an existing account. Change the
  password in the UI after the first login and you can remove it from the environment.

## Scripts

Scripts and the CLI keep using the API token (`Authorization: Bearer <API_TOKEN>`); it acts as an admin.
Volunteer workers (`ctip-worker`) authenticate with their own worker tokens and are unaffected.
