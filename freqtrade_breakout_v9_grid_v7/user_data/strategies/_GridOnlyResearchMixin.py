from __future__ import annotations

from typing import Any

from pandas import DataFrame


class GridOnlyResearchMixin:
    """Remove the Breakout sleeve without changing the parent Grid policy."""

    MAX_OPEN_TRADES = 1

    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict[str, Any],
    ) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        tags = dataframe["enter_tag"].fillna("").astype(str)
        non_grid = ~tags.str.startswith("grid_")
        dataframe.loc[non_grid, "enter_long"] = 0
        dataframe.loc[non_grid, "enter_short"] = 0
        dataframe.loc[non_grid, "enter_tag"] = None

        pair = str(metadata.get("pair") or "")
        ranks = getattr(self, "_candidate_ranks", {}).get(pair)
        if pair and ranks is not None:
            self._candidate_ranks[pair] = {
                key: value
                for key, value in ranks.items()
                if key[0] == "grid"
            }
        return dataframe
