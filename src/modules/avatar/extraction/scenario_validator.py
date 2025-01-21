import os
from glob import glob
from itertools import groupby
from operator import itemgetter
from typing import List, Dict, Set

from modules.config.logger import Logger


class ScenarioValidator:
    """
    Validates the existence of scenario directories and frame files within each scenario.
    """

    def __init__(self, root_dir: str) -> None:
        """
        Initializes the ScenarioValidator with a given root directory.

        Args:
            root_dir (str): The root directory containing all scenario subfolders.
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.root_dir = root_dir
        self.logger.info(f"Initialized {self.__class__.__name__} with root_dir='{root_dir}'")

    def verify_scenarios(self, all_scenario_ids) -> List[str]:
        """
        Verifies that scenario directories corresponding to the given scenario IDs exist.

        Args:
            all_scenario_ids (List[str]): A collection of scenario IDs to verify.

        Returns:
            List[str]: A list of valid scenario IDs that exist in the root directory.
        """
        missing_scenarios = []
        valid_scenarios = []

        for scenario_id in all_scenario_ids:
            if self._scenario_exists(scenario_id):
                valid_scenarios.append(scenario_id)
            else:
                # Convert to int for grouping consecutive IDs
                missing_scenarios.append(int(scenario_id))

        if missing_scenarios:
            missing_ranges = self._group_consecutive_ids(sorted(missing_scenarios))
            self._print_missing_scenarios(missing_ranges)
        else:
            self.logger.info("All scenarios exist.")

        return valid_scenarios

    def verify_frames(self, valid_scenario_ids: List[str], all_frame_ids: List[str]) -> List[str]:
        """
        Verifies the existence of frame files within each valid scenario.

        Args:
            valid_scenario_ids (List[str]): List of valid scenario IDs.
            all_frame_ids (List[str]): List of all frame IDs to verify.

        Returns:
            List[str]: Sorted list of valid frame IDs that exist across all valid scenarios.
        """
        missing_frames: Dict[str, List[str]] = {}
        valid_frame_ids: Set[str] = set(all_frame_ids)

        for scenario_id in valid_scenario_ids:
            scenario_dir = os.path.join(self.root_dir, f"scenario_{scenario_id}")
            existing_frames = self._get_existing_frames(scenario_dir, all_frame_ids)
            missing = set(all_frame_ids) - existing_frames

            if missing:
                missing_frames[scenario_id] = sorted(missing)
                valid_frame_ids &= existing_frames
                self.logger.info(f"Scenario {scenario_id}: Missing frames {sorted(missing)}")
            else:
                valid_frame_ids &= existing_frames
                self.logger.info(f"Scenario {scenario_id}: All frames exist.")

        if missing_frames:
            self._print_missing_frames(missing_frames)
        else:
            self.logger.info("All frames exist for the valid scenarios.")

        return sorted(valid_frame_ids)

    def _scenario_exists(self, scenario_id: str) -> bool:
        """
        Checks if a scenario directory exists.

        Args:
            scenario_id (str): Scenario ID to check.

        Returns:
            bool: True if scenario exists, False otherwise.
        """
        scenario_dir = os.path.join(self.root_dir, f"scenario_{scenario_id}")
        return os.path.isdir(scenario_dir)

    @staticmethod
    def _group_consecutive_ids(id_list: List[int]) -> List[str]:
        """
        Groups consecutive integer IDs into ranges.

        Args:
            id_list (List[int]): Sorted list of integer IDs.

        Returns:
            List[str]: List of string representations of grouped ID ranges.
        """
        ranges = []
        for _, g in groupby(enumerate(id_list), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 1:
                ranges.append(f"{str(group[0]).zfill(3)}-{str(group[-1]).zfill(3)}")
            else:
                ranges.append(str(group[0]).zfill(3))
        return ranges

    def _print_missing_scenarios(self, missing_ranges: List[str]) -> None:
        """
        Logs or prints missing scenario ranges.

        Args:
            missing_ranges (List[str]): List of missing scenario ranges.
        """
        msg_intro = (f"Skipping the following scenario(s) because they do not "
                     f"exist in {self.root_dir}:")
        if len(missing_ranges) == 1 and '-' in missing_ranges[0]:
            self.logger.warning(f"{msg_intro}\n{missing_ranges[0]}")
        else:
            self.logger.warning(f"{msg_intro}\n{', '.join(missing_ranges)}")

    @staticmethod
    def _get_existing_frames(scenario_dir: str, frame_ids: List[str]) -> Set[str]:
        """
        Retrieves existing frame IDs within a scenario directory by checking file patterns.

        Args:
            scenario_dir (str): Path to the scenario directory.
            frame_ids (List[str]): List of frame IDs to check.

        Returns:
            Set[str]: A set of existing frame IDs within the scenario directory.
        """
        existing_frames = set()
        for frame_id in frame_ids:
            frame_files = glob(os.path.join(scenario_dir, f"*_{frame_id}.*"))
            if frame_files:
                existing_frames.add(frame_id)
        return existing_frames

    def _print_missing_frames(self, missing_frames: Dict[str, List[str]]) -> None:
        """
        Logs or prints the missing frames for each scenario.

        Args:
            missing_frames (Dict[str, List[str]]): A dictionary mapping scenario IDs to lists of missing frame IDs.
        """
        for scenario_id, frames in missing_frames.items():
            grouped_frames = self._group_consecutive_ids([int(f) for f in frames])
            msg_intro = (f"Skipping the following frame(s) in scenario {scenario_id} "
                         f"because they do not exist:")
            self.logger.warning(f"{msg_intro}\n{', '.join(grouped_frames)}")
