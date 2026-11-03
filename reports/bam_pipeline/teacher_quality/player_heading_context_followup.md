# Deferred player-heading context follow-up

No code change is made while reference v4 runs.

Observed behavior: `_page_heading_context` stores a player card and a later `RECENT GAMES` heading in the same `player_section` slot. On Miami page 11, line 52 (`#14 TYLER HERRO …`) is therefore replaced by line 67 in the context cache. The final expanded context omits line 52.

Important correction: line 67 itself says `TYLER HERRO RECENT GAMES`. The decoder citations include that line, the column header and the rows. They therefore already supply explicit player identity. Jimmy Butler and Adebayo likewise have named recent-game headings. Omitting the redundant card header does **not** demonstrate an attribution or citation-completeness error for these facts. An earlier diagnostic message overstated that implication; reviewers should not require the extra card header merely to satisfy a checklist.

No fix is warranted from this current valid table. If a future actual source demonstrates identity loss under a generic heading, a narrow candidate fix would retain separate slots for the nearest player-card heading and recent-game subsection heading. Preserve the original source IDs, pages and lines; do not invent a player name or alter body text. A new player-card boundary must stop carrying the prior player's card. Do not modify the frozen older recent-table decoder solely to add redundant card citations: its supported format already requires a named recent-game heading.

Conditional regression plan, only if actual source evidence establishes the generic-heading defect and the parent releases the freeze:

- A two-player page whose second card is followed by a generic `RECENT GAMES` heading retains both that card and the subsection beside a cited row.
- A later card boundary prevents the first player's identity from being the only retained identity for second-player evidence.
- The actual MIA page-11 named heading remains sufficient to identify Herro, with unchanged canonical values, citations and source line numbers.
- Season and column-header context remains present; HTML canonical title/heading behavior is unchanged.

This is source-context hardening, not evidence that the surviving Herro/Butler values or identities are wrong.
