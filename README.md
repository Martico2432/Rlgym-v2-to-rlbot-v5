# RLBotRlGymPythonExample

Example of an RLGym bot using the RLBot framework

Wiki: [VirxEC/python-interface/wiki](https://github.com/VirxEC/python-interface/wiki)

## Quick Start

1. Install Python 3.11 or later.
   - `pip install -r requirements.txt`
1. Download `RLBotServer.exe` and place it in the root directory.
   - <https://github.com/RLBot/core>
   - **_Maybe you have to get an older version, but it should be updated up to 2 days after_**
1. Start a match with `python run.py`.



## Changing the bot

- Put the POLICY.pt directly into src/ folder.
- Modify the action parser and obs to match yours. Use rlgym_compat to make it compatible.
- Modify rlbot.toml to modify the match.

## Recomendations

- Try doing a `deterministic = True` bot vs the `deterministic = False` to know if your bot is better as deterministic. Deafult is False.
- Don't use multiple balls unless your obs supports it. This also means don't join if bot doesn't suport more players, it'll crash and the bot won't move.
- If the custom param loading to simplify things doesn't work, manually set the values.