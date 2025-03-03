# RLBotPythonExample

Example of a Python bot using the RLBot framework

Wiki: [VirxEC/python-interface/wiki](https://github.com/VirxEC/python-interface/wiki)

## Quick Start

1. Install Python 3.11 or later
1. Create a Python virtual environment
   - `python3 -m venv venv`
1. Activate the virtual environment
   - Windows: `.\venv\Scripts\activate`
   - Linux: `source venv/bin/activate`
1. Install the required packages
   - `pip install -r requirements.txt`
1. Download `RLBotServer.exe` and place it in the root directory
   - <https://github.com/RLBot/core>
1. Modify `rlbot.toml` to your liking
   - Note: `dev.toml` also exists with a few changed settings that might be useful for development
1. Start a match with `python run.py`

## Changing the bot

- Bot behavior is controlled by `src/bot.py`
- Bot appearance is controlled by `src/loadout.toml`

## Configuring for the v5 botpack

1. `pip install pyinstaller`
1. `pyinstaller --onefile src/bot.py --paths src` -
  This will create a file called `bot.spec`.
1. Create `bob.toml` in the same directory as the spec file with the following content:
   ```toml
   [[config]]
   project_name = "PythonExample"
   bot_configs = ["src/bot.toml"]

   [config.builder_config]
   builder_type = "pyinstaller"
   entry_file = "bot.spec"
   ```

   - `project_name` will be the name of your bot's folder in the botpack
   - `bot_configs` is a list of bot configs that will be included in the botpack
   - `builder_type` should always be `pyinstaller`
   - `entry_file` is the name of the spec file

1. Commit both `bot.spec` and `bob.toml` to your bot's repository.
  Note that `bob.toml` CANNOT be renamed, but `bot.spec` can be anything as long as `entry_file` is also renamed to reflect the change.
