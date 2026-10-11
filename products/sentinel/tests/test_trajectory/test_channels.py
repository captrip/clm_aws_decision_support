import threading
from sentinal.core.execution.state import ExecutionState
from sentinal.core.trajectory import TrajectoryStep, Correction, CorrectionFlag


def _step(i):
    return TrajectoryStep(index=i, kind="agent", summary=f"s{i}", raw={})


def test_append_and_get_trajectory_snapshot():
    st = ExecutionState()
    st.append_step(_step(0))
    snap = st.get_trajectory()
    st.append_step(_step(1))
    assert len(snap) == 1          # snapshot is isolated
    assert len(st.get_trajectory()) == 2


def test_push_and_drain_corrections_once():
    st = ExecutionState()
    st.push_correction(Correction(flag=CorrectionFlag.REDIRECT, note="x"))
    drained = st.drain_corrections()
    assert len(drained) == 1
    assert drained[0].note == "x"
    assert st.drain_corrections() == []   # cleared after drain


def test_channels_thread_safe():
    st = ExecutionState()

    def writer():
        for i in range(100):
            st.append_step(_step(i))
            st.push_correction(Correction(flag=CorrectionFlag.CONTINUE))

    ts = [threading.Thread(target=writer) for _ in range(5)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(st.get_trajectory()) == 500
    assert len(st.drain_corrections()) == 500
