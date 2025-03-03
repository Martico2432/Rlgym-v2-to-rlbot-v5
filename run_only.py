from pathlib import Path

from rlbot.managers import MatchManager

MATCH_CONFIG_PATH = "rlbot.toml"

if __name__ == "__main__":
    root_dir = Path(__file__).parent

    # RLBotServer MUST BE STARTED MANUALLY!
    # ONLY start the match
    match_manager = MatchManager()
    match_manager.start_match(root_dir / MATCH_CONFIG_PATH, False)

    # wait
    input("\nPress any enter to end the match: ")

    # end the match and disconnect, don't shutdown RLBotServer
    match_manager.stop_match()
    match_manager.disconnect()
