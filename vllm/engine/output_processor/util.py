from typing import List
from typing import Sequence as GenericSequence
from typing import cast

from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[SamplerOutput],
        scheduled_seq_groups: List[ScheduledSequenceGroup],
        return_hidden_states: bool = False) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[CompletionSequenceGroupOutput]] = [
        [] for _ in scheduled_seq_groups
    ]
    for step in outputs:
        sequence_group_output: CompletionSequenceGroupOutput
        for i, sequence_group_output in enumerate(step):
            if return_hidden_states and isinstance(step, SamplerOutput):
                assert len(scheduled_seq_groups[i].seq_group.seqs) == 1
                sequence_group_output.hidden_state = (
                    step.hidden_states[i, :].clone().cpu().unsqueeze(0))
                
            output_by_sequence_group[i].append(sequence_group_output)

    # Cast to the more generic type that CompletionSequenceGroupOutput
    # inherits from.
    return cast(List[List[SequenceGroupOutput]], output_by_sequence_group)
