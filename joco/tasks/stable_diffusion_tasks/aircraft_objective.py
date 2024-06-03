from joco.tasks.stable_diffusion_tasks.stable_diffusion_objective import (
    StableDiffusionObjective,
)


class GenerateAircraftsPrepending(StableDiffusionObjective):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

    def set_task_specific_vars(self):
        """Method sets the following three variables for the specific
        prompt optimization task
        1. self.optimal_class: string specifying the optimal/target imagenet class
            (we seek to find prompts that generate images of this class)
        2. self.avoid_class: string specifying the imagenet class that we do not want to generate images of, use None if irrelevant
        3.self.prepend_to_text: string specifying text that we prepend all prompts to, if not a prepending task use empty string
        """
        self.optimal_class = "aircraft"
        self.avoid_class = "shore"
        self.prepend_to_text = "a picture of the ocean"
