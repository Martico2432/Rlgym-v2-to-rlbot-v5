# RLBotRlGymPythonExample

Example of an RLGym bot using the RLBot framework

Wiki: [RLBot/python-interface/wiki](https://github.com/RLBot/python-interface/wiki)

## Setup your RLGym bot

1. Install Python 3.11 or later.
   - Optionally: Create a virtual environment.
1. Run `pip install -r requirements.txt`
1. Put your `POLICY.pt` directly into `src/` folder.
1. Modify the action parser in `act.py` and obs in `obs.py` to match yours.
   - Use the types from [rlgym_compat](https://github.com/JPK314/rlgym-compat) to make it compatible.
   - Update `bot.py` to import and use your act and obs with the right settings.
1. In case you are using a continous actions you also need to replace `DiscreteFF` from `discrete.py` with [ContinuousPolicy](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/continuous_policy.py).
1. Optionally: Update `src/bot.toml`
   - Here you can update the name of the bot and more.
   - If you are using a virtual environment, make sure to update the `run_command` to `..\\venv\\Scripts\\python.exe bot.py`.
   - If you update the `agent_id`, make sure it matches the id found in `src/bot.py`.

## Running your bot in RLBot

1. Download RLBot5 from www.rlbot.org.
1. Run RLBot5 Launcher from the Windows start menu.
1. In the RLBot GUI, click the "Add" button top left and add your `src/bot.toml` in the file picker.
1. Put your bot on a team and click "Start Match".

## Recomendations

- Try doing a `deterministic = True` bot vs the `deterministic = False` to know if your bot is better as deterministic. Default is False.
- Don't use multiple balls unless your obs supports it. This also means don't join if bot doesn't suport more players, it'll crash and the bot won't move.
- If the custom param loading to simplify things doesn't work, manually set the values.
