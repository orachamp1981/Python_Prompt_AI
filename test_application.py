"""Tests for local QA login and persistent session history."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from prompt_page import (
    _help_text,
    _is_greeting,
    _asks_to_see_sql,
    _respond,
)
from cancellation import CancellationToken, PromptCancelled
from session_manager import (
    create_session,
    delete_session,
    delete_user_history,
    get_user_sessions,
    init_history_db,
    load_messages,
    save_message,
)
from users_authentication import authenticate, can_view_sql
from ollama_client import stream_guest_answer


class AuthenticationTests(unittest.TestCase):
    def test_local_test_accounts_have_expected_roles(self):
        self.assertEqual(authenticate("admin", "123")["role"], "admin")
        self.assertEqual(authenticate("developer", "123")["role"], "developer")
        self.assertEqual(authenticate("user", "123")["role"], "normal_user")
        self.assertIsNone(authenticate("admin", "wrong"))

    def test_environment_password_overrides_local_default(self):
        with patch.dict(os.environ, {"PROMPT_AI_ADMIN_PASSWORD": "private-test-pass"}):
            self.assertIsNone(authenticate("admin", "123"))
            self.assertEqual(
                authenticate("admin", "private-test-pass")["role"],
                "admin",
            )

    def test_only_admin_and_developer_can_view_sql(self):
        self.assertTrue(can_view_sql("admin"))
        self.assertTrue(can_view_sql("developer"))
        self.assertFalse(can_view_sql("normal_user"))
        self.assertFalse(can_view_sql("guest"))


class SessionHistoryTests(unittest.TestCase):
    def test_history_is_persistent_and_isolated_by_user(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.sqlite3")
            with patch.dict(os.environ, {"PROMPT_AI_HISTORY_DB": history_path}):
                init_history_db()
                empty_session = create_session("alice")
                session_id = create_session("alice")
                save_message(session_id, "alice", "user", "Region business August 2026")
                save_message(session_id, "alice", "assistant", "Query plan ready.")

                self.assertEqual(
                    load_messages(session_id, "alice"),
                    [
                        {"role": "user", "content": "Region business August 2026"},
                        {"role": "assistant", "content": "Query plan ready."},
                    ],
                )
                self.assertEqual(load_messages(session_id, "bob"), [])
                sessions = get_user_sessions("alice")
                self.assertEqual(sessions, [(session_id, "Region business August 2026")])
                self.assertNotIn(empty_session, [item[0] for item in sessions])
                with self.assertRaises(LookupError):
                    save_message(session_id, "bob", "user", "unauthorized")

    def test_delete_one_chat_and_clear_history_are_user_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "history.sqlite3")
            with patch.dict(os.environ, {"PROMPT_AI_HISTORY_DB": history_path}):
                init_history_db()
                alice_first = create_session("alice")
                alice_second = create_session("alice")
                bob_session = create_session("bob")
                for session_id, owner in (
                    (alice_first, "alice"),
                    (alice_second, "alice"),
                    (bob_session, "bob"),
                ):
                    save_message(session_id, owner, "user", "hello")

                delete_session(alice_first, "alice")
                self.assertEqual(
                    [item[0] for item in get_user_sessions("alice")],
                    [alice_second],
                )
                with self.assertRaises(LookupError):
                    delete_session(bob_session, "alice")

                delete_user_history("alice")
                self.assertEqual(get_user_sessions("alice"), [])
                self.assertEqual(
                    [item[0] for item in get_user_sessions("bob")],
                    [bob_session],
                )


class PromptHelpTests(unittest.TestCase):
    def test_greetings_are_handled_without_business_metadata(self):
        self.assertTrue(_is_greeting("Hi!"))
        self.assertIn("returns matching business results", _help_text("normal_user"))

    def test_non_query_intent_is_answered_by_the_metadata_grounded_planner(self):
        metadata = type(
            "Metadata",
            (),
            {"field_catalog": lambda _self: {}},
        )()
        informational_answer = (
            "This package supports business summaries, case counts, registration "
            "year grouping, policy counts, and policy lookups by CNIC."
        )
        with patch(
            "prompt_page.read_package_metadata",
            return_value=metadata,
        ), patch(
            "prompt_page.interpret_prompt",
            return_value={
                "status": "informational",
                "message": informational_answer,
                "measure_field": None,
                "group_by": [],
            },
        ) as planner:
            response, result = _respond(
                "Explain the kinds of information this workspace can return.",
                "normal_user",
            )
        self.assertEqual(response, informational_answer)
        self.assertEqual(result["status"], "informational")
        planner.assert_called_once_with(
            "Explain the kinds of information this workspace can return.",
            metadata,
        )

    def test_sql_is_shown_only_when_explicitly_requested(self):
        self.assertTrue(_asks_to_see_sql("Show me the SQL for this result"))
        self.assertTrue(_asks_to_see_sql("Can you display the query?"))
        self.assertFalse(_asks_to_see_sql("Total region-wise business for August 2026"))
        self.assertFalse(_asks_to_see_sql("What is the business?"))

    def test_clarification_hides_internal_measure_aliases(self):
        metadata = type(
            "Metadata",
            (),
            {
                "field_catalog": lambda _self: {
                    "RECEIVED_AMOUNT": {
                        "name": "RECEIVED_AMOUNT",
                        "labels": {
                            "Received Amount",
                            "Business",
                            "Topup",
                        },
                    }
                }
            },
        )()
        with patch("prompt_page.read_package_metadata", return_value=metadata), patch(
            "prompt_page.interpret_prompt",
            return_value={
                "status": "clarify",
                "measure_field": "RECEIVED_AMOUNT",
                "group_by": [],
                "message": (
                    "Please confirm: sum of 'Received Amount' or a different field?"
                ),
            },
        ):
            response, _result = _respond(
                "Top up for September 2026",
                "normal_user",
            )

        self.assertNotIn("Received Amount", response)
        self.assertIn("Topup", response)

    def test_every_guest_message_is_sent_to_ollama_with_chat_history(self):
        history = [
            {"role": "user", "content": "What is SQL?"},
            {"role": "assistant", "content": "SQL queries relational databases."},
            {"role": "user", "content": "Show me a simple example."},
        ]
        for prompt in ("Hi", "What is prompt for?", "You know about SQL?"):
            with self.subTest(prompt=prompt), patch(
                "prompt_page.stream_guest_answer",
                return_value=iter(["Ollama response"]),
            ) as ollama:
                response, result = _respond(prompt, "guest", history)

                self.assertEqual(response, "Ollama response")
                self.assertEqual(result, {"kind": "guest"})
                ollama.assert_called_once_with(history)

    def test_guest_stream_sends_history_and_yields_incremental_content(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                chunks = ("Hello", " there")
                for chunk in chunks:
                    yield json.dumps(
                        {"message": {"content": chunk}, "done": False}
                    ).encode("utf-8") + b"\n"
                yield b'{"message":{"content":""},"done":true}\n'

        messages = [{"role": "user", "content": "Hi"}]
        with patch("ollama_client.urlopen", return_value=FakeResponse()) as open_url:
            self.assertEqual(list(stream_guest_answer(messages)), ["Hello", " there"])

        request = open_url.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen2.5:1.5b")
        self.assertTrue(body["stream"])
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["options"]["num_predict"], 512)

    def test_guest_stream_closes_ollama_response_when_cancelled(self):
        cancellation = CancellationToken()

        class FakeResponse:
            closed = False
            line_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def close(self):
                self.closed = True

            def __iter__(self):
                yield b'{"message":{"content":"Starting"}}\n'
                cancellation.cancel()
                yield b'{"message":{"content":"more"}}\n'

        response = FakeResponse()
        with patch("ollama_client.urlopen", return_value=response):
            with self.assertRaises(PromptCancelled):
                list(
                    stream_guest_answer(
                        [{"role": "user", "content": "A long question"}],
                        cancellation,
                    )
                )

        self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
