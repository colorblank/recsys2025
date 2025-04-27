import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
from pathlib import Path
import argparse
import logging


from data_utils.constants import (
    EventTypes,
    DAYS_IN_TARGET,
)
from data_utils.utils import join_properties
from data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class DataSplitter:
    """数据分割器类，用于处理推荐系统竞赛数据的时间序列划分。

    该类负责将事件数据按时间顺序分割为训练输入、训练目标和验证目标三个部分。
    训练输入包含截止到训练目标开始日期的所有事件数据；训练目标和验证目标分别包含
    后续时间窗口内的事件数据。对于购买事件，还会关联产品属性信息。

    属性:
        challenge_data_dir (DataDir): 竞赛数据目录对象，包含原始事件数据、输入和目标文件夹的路径
        days_in_target (int): 目标时间窗口的天数
        end_date (datetime): 数据集的结束日期
        input_events (Dict[str, pd.DataFrame]): 存储各类事件的训练输入数据
        target_events (Dict[str, pd.DataFrame]): 存储训练和验证的目标数据
    """

    def __init__(
        self,
        challenge_data_dir: DataDir,
        days_in_target: int,
        end_date: datetime,
    ) -> None:
        """初始化数据分割器。

        参数:
            challenge_data_dir (DataDir): 竞赛数据目录对象，包含数据文件路径信息
            days_in_target (int): 目标时间窗口的天数
            end_date (datetime): 数据集的结束日期，通常为原始数据中最后一个事件的时间
        """
        self.challenge_data_dir = challenge_data_dir
        self.days_in_target = days_in_target
        self.end_date = pd.to_datetime(end_date)

        self.input_events: Dict[str, pd.DataFrame] = {}
        self.target_events: Dict[str, pd.DataFrame] = {}

    def _compute_target_start_dates(self) -> Tuple[datetime, datetime]:
        """计算训练和验证目标的起始日期。

        从结束日期开始，减去两个目标时间窗口（训练和验证）再减一天，以确保从每天的
        00:00:00开始计算，而不是从结束日期往前推24小时。

        返回:
            Tuple[datetime, datetime]: 包含训练目标起始日期和验证目标起始日期的元组
        """
        train_target_start = self.end_date - timedelta(days=2 * self.days_in_target - 1)
        train_target_start = train_target_start.replace(hour=0, minute=0, second=0)
        validation_target_start = train_target_start + timedelta(self.days_in_target)
        return train_target_start, validation_target_start

    def _create_input_chunk(
        self,
        event_df: pd.DataFrame,
        train_target_start: datetime,
    ) -> pd.DataFrame:
        """创建训练输入数据块。

        提取指定日期之前的所有事件作为训练输入数据。

        参数:
            event_df (pd.DataFrame): 包含所有事件的数据框
            train_target_start (datetime): 训练目标的起始日期

        返回:
            pd.DataFrame: 训练输入数据框
        """
        train_input = event_df.loc[event_df["timestamp"] < train_target_start]
        return train_input

    def _create_target_chunks(
        self,
        event_df: pd.DataFrame,
        properties_df: pd.DataFrame,
        train_target_start: datetime,
        validation_target_start: datetime,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """创建训练和验证目标数据块。

        将事件数据分割为训练目标和验证目标两部分，并关联产品属性信息。训练目标包含从
        训练起始日期到验证起始日期之间的事件，验证目标包含从验证起始日期到结束日期之间的事件。

        参数:
            event_df (pd.DataFrame): 包含所有事件的数据框
            properties_df (pd.DataFrame): 产品属性数据框
            train_target_start (datetime): 训练目标的起始日期
            validation_target_start (datetime): 验证目标的起始日期

        返回:
            Tuple[pd.DataFrame, pd.DataFrame]: 包含训练目标数据框和验证目标数据框的元组
        """
        train_target = event_df.loc[
            (event_df["timestamp"] >= train_target_start)
            & (event_df["timestamp"] < validation_target_start)
        ]
        validation_target = event_df.loc[
            (event_df["timestamp"] >= validation_target_start)
            & (event_df["timestamp"] <= self.end_date)
        ]

        train_target = join_properties(train_target, properties_df)
        validation_target = join_properties(validation_target, properties_df)

        return train_target, validation_target

    def split(self) -> None:
        """执行数据分割操作。

        将事件数据按时间顺序分割为三个部分：
        1. 训练输入数据：包含截止到训练目标开始日期的所有事件
        2. 训练目标数据：包含训练目标时间窗口内的事件
        3. 验证目标数据：包含验证目标时间窗口内的事件

        对于购买事件，还会关联产品属性信息到目标数据中。
        """
        train_target_start, validation_target_start = self._compute_target_start_dates()

        for event_type in EventTypes:
            msg = f"Creating splits for {event_type.value} event type"
            logger.info(msg=msg)
            events = self.load_events(event_type=event_type)
            events["timestamp"] = pd.to_datetime(events.timestamp)

            train_input = self._create_input_chunk(
                event_df=events, train_target_start=train_target_start
            )
            self.input_events[event_type.value] = train_input

            if event_type == "product_buy":
                properties = pd.read_parquet(self.challenge_data_dir.properties_file)
                train_target, validation_target = self._create_target_chunks(
                    event_df=events,
                    properties_df=properties,
                    train_target_start=train_target_start,
                    validation_target_start=validation_target_start,
                )
                self.target_events["train_target"] = train_target
                self.target_events["validation_target"] = validation_target

    def save_splits(self) -> None:
        """保存分割后的数据。

        将分割后的训练输入数据和目标数据保存到竞赛数据文件夹的相应子目录中。
        训练输入数据保存在input目录下，目标数据保存在target目录下。
        """
        for event_type, events in self.input_events.items():
            msg = f"Saving {event_type} train input"
            logger.info(msg=msg)
            events.to_parquet(
                self.challenge_data_dir.input_dir / f"{event_type}.parquet", index=False
            )

        for target_type, events in self.target_events.items():
            msg = f"Saving {target_type}"
            logger.info(msg=msg)
            events.to_parquet(
                self.challenge_data_dir.target_dir / f"{target_type}.parquet",
                index=False,
            )

    def load_events(self, event_type: EventTypes) -> pd.DataFrame:
        """加载指定类型的事件数据。

        参数:
            event_type (EventTypes): 事件类型

        返回:
            pd.DataFrame: 包含指定类型事件的数据框
        """
        return pd.read_parquet(
            self.challenge_data_dir.data_dir / f"{event_type.value}.parquet"
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge-data-dir",
        type=str,
        required=True,
        help="Competition data directory which should consists of event files, product properties and two subdirectories — input and target",
    )
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    challenge_data_dir = DataDir(data_dir=Path(params.challenge_data_dir))

    product_buy = pd.read_parquet(
        challenge_data_dir.data_dir / f"{EventTypes.PRODUCT_BUY.value}.parquet"
    )
    end_date = pd.to_datetime(product_buy["timestamp"].max())

    splitter = DataSplitter(
        challenge_data_dir=challenge_data_dir,
        days_in_target=DAYS_IN_TARGET,
        end_date=end_date,
    )
    splitter.split()
    splitter.save_splits()


if __name__ == "__main__":
    main()
