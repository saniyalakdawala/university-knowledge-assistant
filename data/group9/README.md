# Group 9 hand-off: the new course's syllabus passages

Put Group 9's file here as `group4_chunks.csv` (columns: `chunk_id, page, section, part, text`).
It comes from Group 9's `POST /modernize-curriculum` response (`handoffs.group4_chunks`) or their
`data/week7/group4_chunks.csv`. The server loads it at start-up.

You can also load it while the server runs, without restarting:

```bash
curl -X POST http://localhost:8000/api/knowledge/course -H "Content-Type: application/json" -d @group9_modernize_response.json
```

`tests/fixtures/sample_group9_chunks.csv` is a made-up sample in the same format, used only by the tests.
