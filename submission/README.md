# Selected Kaggle Final Submission records

Include:

1. a completed `submission_info.yaml` listing every submission marked `Selected` on Kaggle for this track, up to two;
2. `final_submission_1.csv`, downloaded from Kaggle using the first Submission ID; and
3. `final_submission_2.csv`, only when a second submission was Selected.

For each ID, use `kaggle competitions submission-download <SUBMISSION_ID>` to retrieve the exact submitted file. You may rename the downloaded CSV to the fixed name above, but do not open and resave, reorder, normalize, or otherwise rewrite its contents. Record the resulting SHA-256 in `submission_info.yaml`.

Enter as `primary_verification_submission_id` the Selected submission with the higher private score — the one that determined the team's official final private-leaderboard ranking and Top 15 qualification. The Organizing Committee will confirm it against Kaggle's records. Map every Selected ID to the exact model, checkpoint, and configuration needed to reproduce it.

The primary verification submission will always be run. A second Selected submission normally receives static audit only, but its complete materials and runnable mapping are still required in case the Organizing Committee escalates the check.
