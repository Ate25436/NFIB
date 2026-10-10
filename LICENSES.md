# Licenses and attribution

This repository combines source text, AMALGUM annotation layers, newly created
negation-focus annotations, and software. These components do not share one
blanket license.

## Source text by genre

| Dataset directory | Source | Text license | License text |
| --- | --- | --- | --- |
| `data/biography/` | Wikipedia | CC BY-SA 3.0 Unported | [`LICENSE.txt`](data/biography/LICENSE.txt) |
| `data/how_to_guide/` | wikiHow | CC BY-NC-SA 3.0 Unported | [`LICENSE.txt`](data/how_to_guide/LICENSE.txt) |
| `data/interview/` | Wikinews | CC BY 2.5 Generic | [`LICENSE.txt`](data/interview/LICENSE.txt) |
| `data/news/` | Wikinews | CC BY 2.5 Generic | [`LICENSE.txt`](data/news/LICENSE.txt) |
| `data/travel_guide/` | Wikivoyage | CC BY-SA 3.0 Unported | [`LICENSE.txt`](data/travel_guide/LICENSE.txt) |

The `source.text_license` object in every record identifies the applicable
license. The `source.amalgum_metadata` object supplies the source page URL and
title needed for attribution. Preserve these fields and the corresponding
`LICENSE.txt` when redistributing the data.

Important consequences include:

- wikiHow-derived records are licensed for non-commercial use under
  CC BY-NC-SA 3.0;
- ShareAlike requirements apply to adaptations of Wikipedia-, wikiHow-, and
  Wikivoyage-derived material; and
- attribution requirements apply to all five genres.

The linked license text controls if this summary differs from it.

## AMALGUM annotations

The AMALGUM linguistic annotation layers are identified in each record by
`source.amalgum_annotation_license` as
[CC BY 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The upstream corpus is available from
[gucorpling/amalgum](https://github.com/gucorpling/amalgum). Released records
were matched against revision
`a2dbd33e757f75a61d68d22d1a29e840bdffa711`.

## Negation-focus labels and software

The newly created negation-focus `label` and `votes` fields and the original
code in this repository are licensed under
[Creative Commons Attribution 4.0 International](LICENSE.txt). Reusers must
give appropriate credit, link to the license, and indicate whether changes
were made.

This CC BY 4.0 grant applies only to those newly created labels and original
code. It does not relicense the source text or AMALGUM annotation layers, which
remain governed by the licenses described above.

## No legal advice

This file summarizes the provenance information shipped with the dataset and is
not legal advice. Users are responsible for determining whether their intended
use satisfies all applicable licenses and laws.
