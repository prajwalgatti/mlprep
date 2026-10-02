# ML interview prep app for iPhone

An offline iOS app for ML/AI interview prep. It has explainers, quizzes,
flashcards with spaced repetition (FSRS), and mock exams. No account or server needed.

## 1. Download
You need a Mac with Xcode 16+ and an iPhone on iOS 17+.
```bash
git clone https://github.com/prajwalgatti/mlprep.git
```

## 2. Install on your iPhone
1. Create your local signing config:
   ```bash
   cp Config/Local.xcconfig.example Config/Local.xcconfig
   ```
   Then edit the two lines in it. Set `DEVELOPMENT_TEAM` to your Team ID. To find it, sign in under Xcode → Settings → Accounts,
   then look under *Signing & Capabilities* → Personal Team. Set `BUNDLE_ID_PREFIX` to anything unique, e.g. `io.github.<you>`.
2. Open `PrepApp.xcodeproj`, select your plugged-in iPhone, and press **⌘R**.
3. First time only, on the phone, turn on Settings → Privacy & Security → **Developer Mode**.
   Then trust your certificate under Settings → General → VPN & Device Management.

With a free Apple ID the app expires after 7 days. Press ⌘R again to renew it; your progress is kept.
To just try the app, pick a simulator instead of your phone.

## 3. Add content
Each topic is one YAML file in `PrepApp/Content/<area>/`, and the app picks up new files automatically.
1. Write or edit a topic following [CONTENT_GUIDE.md](CONTENT_GUIDE.md).
2. Check it with the validator, then open a PR:
   ```bash
   python3 tools/validate.py
   ```
   The validator needs PyYAML (`pip3 install pyyaml`).

**With an AI agent:** point it at [AGENTS.md](AGENTS.md), for example: *"Add a topic on speculative decoding, following AGENTS.md."*
To report mistakes, flag them in the app, then go to Saved → Flagged → Export and paste the list into your agent or an issue.
