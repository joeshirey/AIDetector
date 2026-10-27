So we finally ripped out our old metrics stack last month. Took way longer than I thought it would. Three weeks, give or take, and most of that was arguing about label names.

The thing nobody tells you about migrating dashboards is that half of them are dead. We had 212. I counted. Maybe forty get looked at in any given week, and a dozen of those are the on-call board plus its weird cousins that somebody forked in 2021 and never cleaned up.

Anyway. The plan was to port everything over as-is and fix it later, which is what you always say and never do. Mark talked us out of that, thank god. We sat down with the on-call rotation and asked what they actually open when the pager goes off at 3am. The answer was depressing: mostly the logs, and a single latency graph that's been broken since the last Postgres upgrade because the query hardcodes a hostname.

I'm not going to pretend we did this well. We broke alerting for about six hours on a Tuesday. Nobody noticed, which says something about our alerting too, I guess.

What I'd do differently: delete first, migrate second. If no one screams within a sprint, it's gone. We got a couple of complaints and restored two boards out of the archive. Two. Out of 170-ish deletions.

Next up is tracing, which I'm dreading for completely different reasons, mostly involving a Java service that predates everyone on the team.
