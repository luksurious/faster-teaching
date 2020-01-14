# Notes:
- Belief update questions
- Belief update does not scale in planning


# Planning time

    # New data:
    # - 3*6
    #   448.44 s (observation sampling)
    #   180.21 s (all observations and skip 0; no epsilon)
    #   344.21 s (all observations, with epsilon) - no convergence - deteriorated to quizzes

    # with 10 samples
    # 3: 2s
    # 4: 17s
    # 5: 190s (~*10)
    # 6: ~30min
    # 7: ~5h
    # 8: ~2d
    # 9: ~20d

    # with 9 samples
    # 3: 1.4s
    # 4: 12s
    # 5: 117s (~*9)
    # 6: ~18min
    # 7: ~2.6h
    # 8: ~1d
    # 9: ~10d

    # with 8 samples
    # 3: 0.9s
    # 4: 7.3s
    # 5: 68s (~*8)
    # 6: ~8min
    # 7: ~1h
    # 8: ~8h
    # 9: ~3d

    # with 7 samples
    # 3: 0.6s
    # 4: 4.5s
    # 5: 33s (~*7)
    # 6: ~3.5min
    # 7: ~25min
    # 8: ~3h
    # 9: ~21h

    # with 6 samples
    # 3: 0.5s
    # 4: 2.5s
    # 5: 14s (~*6)
    # 6: ~1.2min
    # 7: ~7min
    # 8: ~42min
    # 9: ~4h