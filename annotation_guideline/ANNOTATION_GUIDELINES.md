# Logically Correct Sentence Selection

You will be given a context and several answer options in the following format:

```
Context:
[sentences]

Options:
a. [sentences]
b. [sentences]
...
x. None of the above
```

## Task

Choose a logically correct option based on the context.

If more than one option is logically correct, choose the option that is closest
to the context.

If no option is logically correct, choose `x` (None of the above).

An option is not treated as logically correct if either of the following is
true:

- The option conflicts with the context.
- The context provides insufficient information to determine whether the
  option is logically correct.

When judging each option, rely solely on information included in the context,
not on common knowledge.

## Example 1

```
Context:
John came to the party to celebrate Mike's birthday.
He didn't come to the party by car.
His car had a broken engine and couldn't be driven.

Options:
a.
John came to the party to celebrate Mike's birthday.
Someone came to the party by car, but not him.
His car had a broken engine and couldn't be driven.
b.
John came to the party to celebrate Mike's birthday.
He came somewhere by car, but not to the party.
His car had a broken engine and couldn't be driven.
c.
John came to the party to celebrate Mike's birthday.
He came to the party, but not by car.
His car had a broken engine and couldn't be driven.
x. None of the above

Answer:
c
```

Option `a` is not logically correct because the context does not explicitly
mention any means of transportation used by another person.

Option `b` is not logically correct because it conflicts with the statement
that John's car could not be driven.

Option `c` is logically correct because the context states that John came to
the party and did not come by car.

Therefore, the answer is `c`.

## Example 2

```
Context:
The FBI initially showed little interest , and he had the impression other federal security agencies were tangled up in legal red tape .
The CIA told him it does not do domestic counterespionage  .
One learns a lot from this book , or seems to , about crippling federal bureaucracy .

Options:
a.
The FBI initially showed little interest , and he had the impression other federal security agencies were tangled up in legal red tape .
The CIA told him that something carries out domestic counterespionage, but it does not.
One learns a lot from this book , or seems to , about crippling federal bureaucracy .
b.
The FBI initially showed little interest , and he had the impression other federal security agencies were tangled up in legal red tape .
The CIA told him it does that, but not domestic counterespionage.
One learns a lot from this book , or seems to , about crippling federal bureaucracy .
x. None of the above

Answer:
x
```

Option `a` is not logically correct because the context does not mention an
organization other than the CIA that conducts domestic counterespionage. The
fact that the FBI conducts domestic counterespionage is common knowledge and is
not included in the context, so it must not be used.

Option `b` is not logically correct because the context does not state that the
CIA performs some other activity. Such outside knowledge must not be used.

Therefore, the answer is `x`.
