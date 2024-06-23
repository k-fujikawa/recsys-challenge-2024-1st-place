import os
import requests

import lightning.pytorch as L
import lightning.pytorch.callbacks
import tenacity
from loguru import logger


class TQDMProgressBarEx(lightning.pytorch.callbacks.TQDMProgressBar):
    def on_validation_batch_end(self, *args, **kwargs):
        super().on_validation_batch_end(*args, **kwargs)
        self.val_progress_bar.set_postfix(
            self.get_metrics(self.trainer, self.trainer.lightning_module)
        )


class SlackNotificationCallback(L.Callback):
    def __init__(
        self,
        exp_id: str,
        channel: str | None = os.getenv("SLACK_CHANNEL"),
        oauth_token: str | None = os.getenv("SLACK_OAUTH_TOKEN"),
        enable: bool = True,
    ):
        self.exp_id = exp_id
        self.channel = channel
        self.oauth_token = oauth_token
        self.enable = enable and (channel is not None) and (oauth_token is not None)
        self.thread_ts = None
        if not self.enable:
            logger.warning("SlackNotificationCallback is disabled.")

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=5, exp_base=3),
        after=tenacity.after_log(logger, "INFO"),  # type: ignore
        stop=tenacity.stop_after_attempt(3),
        reraise=False,
    )
    def _post_to_slack(
        self,
        endpoint: str,
        message: str,
        thread_ts: str | None = None,
    ):
        assert endpoint in ["chat.postMessage", "chat.update"]
        url = f"https://slack.com/api/{endpoint}"
        headers = {"Authorization": f"Bearer {self.oauth_token}"}
        data = {"channel": self.channel, "text": message}
        if thread_ts is not None and endpoint == "chat.postMessage":
            data["thread_ts"] = thread_ts
        elif thread_ts is not None and endpoint == "chat.update":
            data["ts"] = thread_ts
        response = requests.post(url=url, headers=headers, data=data)
        return response.json()

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):
        if not self.enable:
            return
        assert self.channel is not None
        assert self.oauth_token is not None
        response = self._post_to_slack(
            endpoint="chat.postMessage",
            message=f"*{self.exp_id}* @ {os.uname().nodename}\n"
            + "Training in progress... :runner:",
        )
        self.thread_ts = response.get("ts")

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):
        if not self.enable or trainer.global_step == 0:
            return
        assert self.channel is not None
        assert self.oauth_token is not None

        epoch = trainer.current_epoch + 1
        metrics = trainer.progress_bar_metrics
        self._post_to_slack(
            endpoint="chat.update",
            message=f"*{self.exp_id}* @ {os.uname().nodename}\n"
            + f"Epoch end ({epoch} / {trainer.max_epochs})\n"
            + "\n".join([f"・{k}: {v:.4f}" for k, v in metrics.items()]),
            thread_ts=self.thread_ts,
        )
        self._post_to_slack(
            endpoint="chat.postMessage",
            message=f"*{self.exp_id}* @ {os.uname().nodename}\n"
            + f"Training in progress... :runner: (epoch={epoch})\n"
            + "\n".join([f"・{k}: {v:.4f}" for k, v in metrics.items()]),
            thread_ts=self.thread_ts,
        )

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):
        if not self.enable or trainer.global_step == 0:
            return
        assert self.channel is not None
        assert self.oauth_token is not None
        metrics = trainer.progress_bar_metrics
        self._post_to_slack(
            endpoint="chat.update",
            message=f"*{self.exp_id}* @ {os.uname().nodename}\n"
            + "Training completed :white_check_mark:\n"
            + "\n".join([f"・{k}: {v:.4f}" for k, v in metrics.items()]),
            thread_ts=self.thread_ts,
        )
