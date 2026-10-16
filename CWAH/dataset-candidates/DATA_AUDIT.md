# CWAH task-data audit — 2026-10-15

## Verdict

The downloaded WAH dataset is a valid *format candidate* for the CWAH loader,
but it is **not compatible with the requested CollabBench `26_prepare_food`
baseline**. Episode/index 26 is `26_put_dishwasher`, not `26_prepare_food`.
The full CollabBench task dataset was not found in the public source tree. Do
not label a run using this WAH file as a strict CollabBench reproduction.
The requested baseline was not started.

## Existing local data

`CWAH-MultiPlayer/dataset/` contains:

| File | Bytes | Structure / provenance |
| --- | ---: | --- |
| `all_object.py` | 704 | Tracked Python asset; imported with TRACE baseline |
| `env_to_pred.json` | 14,728 | 100-key dictionary; imported with TRACE baseline |
| `match_demo_test.json` | 45,607 | 100-key dictionary; imported with TRACE baseline |
| `object_info.json` | 4,046 | 3-key dictionary; imported with TRACE baseline |
| `object_info_small.json` | 1,040 | 4-key dictionary; imported with TRACE baseline |
| `watch_scenes_split.json` | 4,681,237 | 4-key dictionary (`train`, `test`, `train_pred`, `test_pred`); imported with TRACE baseline |
| `test_env_set_help.pik` | 115,277 | Ignored local file, not in Git; exact origin unverified |

The local pickle SHA-256 is
`6f3b544ebd06d877ffd86a9f2a3da05297da4ef424152d3c0f1bef3f689194e5`.
It contains one list entry: index/task ID 0, `setup_table`, environment 3,
goal `on_plate_kitchentable:2, on_cutleryfork_kitchentable:2`.
The `vh_init_gen_test.py` generator writes to precisely this path and supports
single-task generation, so a smoke-test artifact is plausible, but no
provenance log proves that it generated this exact file. It was left untouched.

## WAH candidate

Downloaded without overwriting local data from
`https://raw.githubusercontent.com/xavierpuigf/watch_and_help/main/dataset/test_env_set_help.pik`
to `CWAH/dataset-candidates/wah/test_env_set_help.pik`.
Upstream WAH commit: `21f5ab7416111d22a2ebc8d0d2427c799d01398a`;
downloaded file Git blob SHA: `d00f87a003b0c2c63a288cd879da7f34cdb12cfb`
(matches the upstream tree). Size: 15,197,562 bytes. SHA-256:
`11992fddda2ef5c9e98d9d19fc7bc425c68f6cfac16dfdbb6832e909a93de1d7`.

Before deserialization, all pickle opcodes were inspected: no `GLOBAL`,
`STACK_GLOBAL`, `REDUCE`, `BUILD`, object-instantiation, extension, or persistent
ID opcodes occur. Data were then parsed with system Python `-I -S` inside a
`bwrap` process with only system libraries and the read-only data file mounted,
and an empty environment. A network namespace was unavailable on this host;
the opcode check, restricted mounts, and isolated Python startup are the
controls used here. Do not generalize this procedure to arbitrary pickles.

It is a list of 100 episodes with sequential, unique task IDs 0–99, two
environments (IDs 3 and 6), and 20 episodes each of `read_book`,
`put_dishwasher`, `prepare_food`, `put_fridge`, and `setup_table`.
Every record has the fields `task_id`, `task_name`, `env_id`, `init_graph`,
`task_goal`, `goal_class`, `level`, `init_rooms`, `pred_str`; the graph has
`nodes` and `edges`, and both agent goals (keys 0 and 1) are present.
The complete index, ID, name, environment ID, and positive-goal summary for
**every episode** are recorded in `wah-manifest.jsonl`. Zero-count predicates
are omitted from each summary; the source pickle is preserved unmodified.

Notable checks:

| Index / task ID | Candidate name | Environment | Profile key in CollabBench |
| ---: | --- | ---: | --- |
| 0 | `read_book` | 3 | `0_read_book` exists |
| 26 | `put_dishwasher` | 3 | `26_prepare_food` exists, but does **not** match |
| 40–59 | `prepare_food` | 3 or 6 | IDs differ from the target 26 |

## Evaluator compatibility and Task 0 cause

`testing_agents/test_symbolic_LLMs.py` loads the pickle as a list. With
`--num-per-task 10` (its default), it runs **all list indices** regardless of
`--test_task`. With another value (the helper script defaults to 2), it uses
`--test_task` as a **list index**, not `task_id`. The argument default is `[0]`.
The environment reset also takes a list index and then reads the record's
`task_id`, `task_name`, `env_id`, graph, rooms and goals. This explains why
the one-record local file only exposes task 0; both dataset size and selector
defaults matter. In the WAH file, index and task ID happen to coincide, but
that must not be assumed for an author-modified dataset.

The WAH candidate satisfies the loader's required field shape, so it is
**structurally compatible**. It is **not semantically compatible** with the
requested CollabBench experiment: `LLM/LLM.py` forms a profile key
`{task_id}_{task_name}_{cluster}` and silently substitutes an empty profile
when no key exists. `cwah.csv` contains `26_prepare_food` for clusters 17,
24, 8, and 23, not `26_put_dishwasher`. Across 100 WAH episodes, only two
task ID/name pairs are present in the CollabBench profile table. A WAH run
would therefore test different tasks and often remove the intended teammate
profile without an error. This is not repaired by relabeling or reindexing.

## Search for author data

The official `HQian-AI/CollabBench` public Git HEAD examined was
`9cb8ca8d28cafe5b17d48b3ed5a5e82be9d88a66`. Its Running and Training
CWAH `dataset/` trees contain the same auxiliary files but **no task pickle**.
The public repository has one commit with no parent and no Git tags, so its
visible Git history and tags do not contain a prior task-pickle release.
The CWAH README lists a dataset directory but supplies no dataset download
URL. No independently verifiable author-modified version was found in the
checked source tree or README. This does not prove that no private or
separately distributed dataset exists.

## Decision

Keep the WAH file as an explicitly labeled **WAH candidate dataset** only.
Do not start the requested DeepSeek-V3.1 × DeepSeek-V3.1 task-26 cluster
baseline until the CollabBench-compatible dataset is obtained and its
`26_prepare_food` episode, goals, and profile mapping are checked. A separate
WAH-format smoke test could be run later, but it must be labeled as such and
cannot support a strict CollabBench-paper reproduction claim.
