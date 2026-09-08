//! Terminal node controls. Every SSH request is short; node jobs own their lifetime.
use crate::remote::{Controller, Node, Operation, Update};
use crate::theme;
use crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::{Line, Span},
    widgets::{Clear, List, ListState, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;

#[derive(Clone)]
pub enum Screen {
    Error {
        message: String,
        previous: Box<Screen>,
    },
    Nodes,
    Remove(Node),
    Edit {
        node: Node,
        field: usize,
        editing: bool,
    },
    Jobs,
    Job,
    Review {
        operation: Operation,
        review: Value,
    },
    Stop {
        job: String,
    },
    Resume {
        job: String,
        checkpoint: String,
        output: String,
        field: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};
    fn controller() -> Controller {
        let mut c = Controller::load(
            &std::env::temp_dir().join(format!("arwen-ui-test-{}", crate::remote::stamp())),
        );
        let mut node = Node::blank();
        node.host = "weather-node".into();
        node.workspace = "/srv/weather".into();
        node.config = "/srv/weather/storm.toml".into();
        node.last_job = Some("job-1".into());
        c.store.active = Some(node.id.clone());
        c.store.nodes.push(node);
        c.view.jobs = vec![serde_json::json!({"id":"job-1", "state":"completed"})];
        c
    }
    fn press(panel: &mut Panel, code: KeyCode, c: &Controller) -> Intent {
        panel.key(
            KeyEvent::new(code, KeyModifiers::NONE),
            c,
            Ok("none".into()),
        )
    }
    #[test]
    fn all_node_screens_keep_primary_mouse_actions_at_supported_sizes() {
        let c = controller();
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            for (screen, label) in [
                (Screen::Nodes, "Enter Use"),
                (
                    Screen::Edit {
                        node: c.store.nodes[0].clone(),
                        field: 9,
                        editing: true,
                    },
                    "Ctrl+S Save",
                ),
                (Screen::Jobs, "Enter View"),
                (Screen::Remove(c.store.nodes[0].clone()), "Enter Remove profile"),
                (Screen::Job, "X Stop"),
                (
                    Screen::Stop {
                        job: "job-1".into(),
                    },
                    "Enter Stop job",
                ),
                (
                    Screen::Resume {
                        job: "job-1".into(),
                        checkpoint: "latest".into(),
                        output: String::new(),
                        field: 0,
                    },
                    "Enter Review",
                ),
                (
                    Screen::Review {
                        operation: Operation::Start {
                            products: "none".into(),
                            preview: true,
                            binding: None,
                        },
                        review: serde_json::json!({"config":"/storm.toml","outdir":"/new","products":"none"}),
                    },
                    "Enter Start on node",
                ),
            ] {
                let mut p = Panel {
                    screen,
                    ..Panel::default()
                };
                let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
                terminal
                    .draw(|f| p.draw(f, f.area(), &c, "General · 25"))
                    .unwrap();
                let buffer = terminal.backend().buffer();
                let rendered: String = (0..height)
                    .flat_map(|y| (0..width).map(move |x| buffer[(x, y)].symbol()))
                    .collect();
                assert!(rendered.contains(label), "{width}x{height}: {label}");
                assert!(p
                    .hits
                    .iter()
                    .all(|(r, _)| r.bottom() <= height && r.right() <= width));
            }
        }
    }
    #[test]
    fn start_requires_review_and_stop_is_explicit() {
        let c = controller();
        let mut p = Panel::default();
        assert!(matches!(
            press(&mut p, KeyCode::Char('s'), &c),
            Intent::Request(Operation::Start { preview: true, .. })
        ));
        p.screen = Screen::Job;
        assert!(matches!(
            press(&mut p, KeyCode::Char('x'), &c),
            Intent::Keep
        ));
        assert!(matches!(p.screen, Screen::Stop { .. }));
        assert!(matches!(press(&mut p, KeyCode::Esc, &c), Intent::Keep));
        assert!(matches!(p.screen, Screen::Job));
        press(&mut p, KeyCode::Char('x'), &c);
        assert!(matches!(
            press(&mut p, KeyCode::Enter, &c),
            Intent::Request(Operation::Stop { .. })
        ));
    }
    #[test]
    fn removing_a_profile_is_reviewed_and_never_requests_a_remote_stop() {
        let c = controller();
        let original = c.store.nodes[0].clone();
        let mut panel = Panel::default(); panel.open(&c);
        assert!(matches!(press(&mut panel, KeyCode::Delete, &c), Intent::Keep));
        assert!(matches!(&panel.screen, Screen::Remove(node) if node == &original));
        let screen = render(&mut panel, &c, 65, 20);
        assert!(screen.contains("Remote jobs keep running"), "{screen}");
        let mut held = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        held.kind = KeyEventKind::Repeat;
        assert!(matches!(panel.key(held, &c, Ok("none".into())), Intent::Keep));
        assert!(matches!(press(&mut panel, KeyCode::Esc, &c), Intent::Keep));
        assert!(matches!(panel.screen, Screen::Nodes));
        assert_eq!(c.store.nodes, vec![original.clone()]);
        press(&mut panel, KeyCode::Delete, &c);
        assert!(matches!(press(&mut panel, KeyCode::Enter, &c), Intent::Remove(id) if id == original.id));
        assert_eq!(c.store.nodes, vec![original]);
        assert!(c.pending.is_none());
    }
    #[test]
    fn empty_jobs_explain_the_next_step_and_only_request_a_start_preview() {
        let mut c = controller(); c.view.jobs.clear();
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut panel = Panel { screen: Screen::Jobs, ..Panel::default() };
            let screen = render(&mut panel, &c, width, height);
            assert!(screen.contains("No jobs are listed"), "{screen}");
            assert!(screen.contains("Enter Review first start"), "{screen}");
            assert!(screen.contains("L Local computer"), "{screen}");
            for code in [KeyCode::Enter, KeyCode::Char('s')] {
                assert!(matches!(press(&mut panel, code, &c), Intent::Request(Operation::Start { preview: true, binding: None, .. })));
            }
            assert!(c.pending.is_none());
        }
    }
    #[test]
    fn empty_nodes_keep_unavailable_actions_local_and_offer_local_computer() {
        let mut c = controller(); c.store.nodes.clear(); c.store.active = None;
        let mut panel = Panel::default(); panel.open(&c);
        for code in [KeyCode::Char('p'), KeyCode::Char('j'), KeyCode::Char('s'), KeyCode::F(7)] {
            assert!(matches!(press(&mut panel, code, &c), Intent::Keep));
            assert!(matches!(panel.screen, Screen::Nodes));
            assert!(panel.notice.contains("Press N"));
        }
        assert!(matches!(press(&mut panel, KeyCode::Delete, &c), Intent::Keep));
        assert!(panel.notice.contains("cannot be removed"));
        assert!(matches!(press(&mut panel, KeyCode::Char('l'), &c), Intent::Select(None)));
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let screen = render(&mut panel, &c, width, height);
            assert!(screen.contains("L Local computer"), "{screen}");
            assert!(!screen.contains("Jobs continue on the node"), "{screen}");
        }
    }
    #[test]
    fn held_enter_cannot_confirm_a_new_launch_resume_or_stop_review() {
        let c = controller();
        let review = serde_json::json!({
            "outdir":"/new-run", "config_sha256":"a".repeat(64),
            "input_sha256":"b".repeat(64), "checkpoint":"/old/checkpoint.npz",
            "checkpoint_sha256":"c".repeat(64), "checkpoint_set_sha256":"d".repeat(64)
        });
        for screen in [
            Screen::Review {
                operation: Operation::Start {
                    products: "none".into(),
                    preview: true,
                    binding: None,
                },
                review: review.clone(),
            },
            Screen::Review {
                operation: Operation::Resume {
                    job: "job-1".into(),
                    checkpoint: "latest".into(),
                    output: String::new(),
                    preview: true,
                    binding: None,
                },
                review,
            },
            Screen::Stop {
                job: "job-1".into(),
            },
        ] {
            let mut p = Panel {
                screen,
                ..Panel::default()
            };
            for kind in [KeyEventKind::Repeat, KeyEventKind::Release] {
                let mut event = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
                event.kind = kind;
                assert!(matches!(p.key(event, &c, Ok("none".into())), Intent::Keep));
            }
            assert!(matches!(
                press(&mut p, KeyCode::Enter, &c),
                Intent::Request(_)
            ));
        }
    }
    #[test]
    fn clicked_save_preserves_case_and_spaces_and_does_not_start_job() {
        let c = controller();
        let mut p = Panel::default();
        let mut n = c.store.nodes[0].clone();
        n.config.clear();
        p.screen = Screen::Edit {
            node: n,
            field: 4,
            editing: true,
        };
        p.paste("/Weather Runs/Drew's Storm.toml");
        let mut terminal = Terminal::new(TestBackend::new(65, 20)).unwrap();
        terminal.draw(|f| p.draw(f, f.area(), &c, "None")).unwrap();
        let r = p
            .hits
            .iter()
            .find(|(_, k)| *k == KeyCode::Char('S'))
            .unwrap()
            .0;
        let result = p.mouse(
            MouseEvent {
                kind: MouseEventKind::Down(MouseButton::Left),
                column: r.x,
                row: r.y,
                modifiers: KeyModifiers::NONE,
            },
            &c,
            Ok("none".into()),
        );
        let Intent::Save(node) = result else {
            panic!("save button did not save")
        };
        assert_eq!(node.config, "/Weather Runs/Drew's Storm.toml");
        assert_eq!(node.last_job.as_deref(), Some("job-1"));
    }
    fn render(panel: &mut Panel, c: &Controller, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|frame| panel.draw(frame, Rect::new(0, 3, width, height - 3), c, "General · 25"))
            .unwrap();
        let buffer = terminal.backend().buffer();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
    fn capture(name: &str, width: u16, height: u16, screen: &str) {
        if let Some(directory) = std::env::var_os("GPUWM_TUI_SNAPSHOT_DIR") {
            let directory = std::path::PathBuf::from(directory);
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join(format!("{name}-{width}x{height}.txt")),
                screen,
            )
            .unwrap();
        }
    }
    #[test]
    fn long_errors_scroll_to_the_final_remedy_and_return_to_saved_inputs() {
        let c = controller();
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let node = c.store.nodes[0].clone();
            let mut p = Panel::default();
            p.screen = Screen::Edit {
                node: node.clone(),
                field: 11,
                editing: true,
            };
            p.error(format!(
                "{}\nFINAL remedy: fix the WPS path, then review again.",
                (0..90)
                    .map(|i| format!("Diagnostic {i}: {}", "weather path ".repeat(6)))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
            let initial = render(&mut p, &c, width, height);
            assert!(!initial.contains("FINAL remedy"));
            press(&mut p, KeyCode::End, &c);
            let last = render(&mut p, &c, width, height);
            assert!(last.contains("FINAL remedy: fix the WPS path"), "{last}");
            assert!(last.contains("Esc Back"), "{last}");
            capture("nodes-error-final-remedy", width, height, &last);
            let rect = p
                .hits
                .iter()
                .find(|(_, key)| *key == KeyCode::Esc)
                .unwrap()
                .0;
            p.mouse(
                MouseEvent {
                    kind: MouseEventKind::Down(MouseButton::Left),
                    column: rect.x,
                    row: rect.y,
                    modifiers: KeyModifiers::NONE,
                },
                &c,
                Ok("none".into()),
            );
            let Screen::Edit {
                node: restored,
                field,
                editing,
            } = &p.screen
            else {
                panic!("did not restore inputs")
            };
            assert_eq!(restored, &node);
            assert_eq!(*field, 11);
            assert!(*editing);
            assert!(c.pending.is_none());
        }
    }
    #[test]
    fn job_end_follows_new_tail_while_up_and_review_keep_their_position() {
        let mut c = controller();
        c.view.status = Some(serde_json::json!({"id":"job-1","state":"running"}));
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut p = Panel::default();
            p.screen = Screen::Job;
            press(&mut p,KeyCode::Char('g'),&c);
            c.view.log = (0..90).map(|i| format!("Step {i}\n")).collect();
            render(&mut p, &c, width, height);
            press(&mut p, KeyCode::End, &c);
            c.view.log.push_str("LATEST FIRST\n");
            let screen = render(&mut p, &c, width, height);
            assert!(screen.contains("LATEST FIRST"), "{screen}");
            press(&mut p, KeyCode::PageUp, &c);
            let before = p.offset;
            c.view.log.push_str("LATEST SECOND\n");
            let screen = render(&mut p, &c, width, height);
            assert_eq!(p.offset, before);
            assert!(!screen.contains("LATEST SECOND"));
            press(&mut p, KeyCode::End, &c);
            let screen = render(&mut p, &c, width, height);
            assert!(screen.contains("LATEST SECOND"));
            press(&mut p, KeyCode::Up, &c);
            let before = p.offset;
            c.view.log.push_str("LATEST THIRD\n");
            render(&mut p, &c, width, height);
            assert_eq!(p.offset, before);
            press(&mut p, KeyCode::End, &c);
            capture(
                "nodes-log-follow",
                width,
                height,
                &render(&mut p, &c, width, height),
            );
            p.update(&Update::Preview { operation:Operation::Start { products:"none".into(),preview:true,binding:None },
                review:serde_json::json!({"config":"/storm.toml","outdir":"/new","products":"none","details":"x".repeat(4000)}) });
            let screen = render(&mut p, &c, width, height);
            assert_eq!(p.offset, 0);
            assert!(screen.contains("START ON"), "{screen}");
            render(&mut p, &c, width, height);
            assert_eq!(p.offset, 0);
        }
    }
    #[test]
    fn actual_native_progress_is_primary_and_raw_logs_are_a_separate_view(){
        let receipt:Value=serde_json::from_str(include_str!("../tests/fixtures/native-progress-newcastle-2013.json")).unwrap();
        let mut c=controller();let mut status=receipt["native_result"].clone();
        status["state"]=receipt["state"].clone();status["id"]=receipt["job_id"].clone();
        c.view.status=Some(status);c.view.log="RAW EVENT LOG DETAIL".into();
        for(width,height)in[(65,20),(80,24),(120,36)]{
            let mut panel=Panel{screen:Screen::Job,..Panel::default()};
            let screen=render(&mut panel,&c,width,height);
            for expected in ["RUNNING","Rendering images","Steps 360","Sim time","Wall time","45.6× realtime","100.0%","Checkpoint saved","G Raw logs"]{assert!(screen.contains(expected),"{expected}: {screen}");}
            assert!(!screen.contains("RAW EVENT LOG DETAIL"));assert!(!screen.contains("arwen.forecast-progress.v1"));
            capture("nodes-native-progress",width,height,&screen);
            assert!(matches!(press(&mut panel,KeyCode::Char('g'),&c),Intent::Keep));
            let logs=render(&mut panel,&c,width,height);assert!(logs.contains("RAW EVENT LOG DETAIL"));assert!(logs.contains("G Progress"));
            press(&mut panel,KeyCode::Char('g'),&c);assert!(render(&mut panel,&c,width,height).contains("Steps 360"));
            assert!(c.pending.is_none());
        }
        c.view.status.as_mut().unwrap()["state"]=serde_json::json!("completed");
        assert!(job_progress_text(c.view.status.as_ref().unwrap(),true).starts_with("COMPLETED · Finished"));
    }
    #[test]
    fn connected_workspace_restores_saved_job_without_opening_the_job_screen(){
        let mut c=controller();c.view.runtime=Some(serde_json::json!({"capabilities":{}}));
        c.view.last_refresh=Some(std::time::Instant::now());
        let panel=Panel::default();assert!(matches!(panel.screen,Screen::Nodes));
        assert!(!panel.should_refresh(&c));
        assert!(panel.should_refresh_connected(&c,true),"A successful probe must fetch the remembered job immediately");
        assert!(c.pending.is_none());assert_eq!(c.store.selected().unwrap().last_job.as_deref(),Some("job-1"));
        c.view.status=Some(serde_json::json!({"id":"job-1","state":"running"}));c.view.eof=Some(true);
        assert!(!panel.should_refresh_connected(&c,true));
        c.view.last_refresh=Some(std::time::Instant::now()-std::time::Duration::from_secs(6));
        assert!(panel.should_refresh_connected(&c,true),"A connected map must keep getting job status while the TUI shows another screen");
        c.view.last_job_refresh=Some(std::time::Instant::now()-std::time::Duration::from_secs(2));
        c.view.last_refresh=Some(std::time::Instant::now());
        assert!(panel.should_refresh_connected(&c,true),"A just-finished timeline request must not postpone a due job-status update");
        c.view.last_job_refresh=Some(std::time::Instant::now());
        c.view.status=None;c.view.connection_error=Some("temporary SSH failure".into());c.view.last_refresh=Some(std::time::Instant::now());
        assert!(!panel.should_refresh_connected(&c,true),"A failed reconnect must retain the retry backoff");
    }
    #[test]
    fn all_twelve_node_fields_are_reachable_in_both_keyboard_directions() {
        let c = controller();
        let node = c.store.nodes[0].clone();
        assert_eq!(node.fields().len(), 12);
        for (width, height) in [(65, 20), (80, 24), (120, 36)] {
            let mut p = Panel::default();
            p.screen = Screen::Edit {
                node: node.clone(),
                field: 0,
                editing: false,
            };
            for expected in 0..node.fields().len() {
                let Screen::Edit { field, .. } = p.screen else {
                    panic!("edit closed")
                };
                assert_eq!(field, expected);
                let screen = render(&mut p, &c, width, height);
                if expected == 11 {
                    assert!(screen.contains("WPS namelist on node"), "{screen}");
                }
                press(&mut p, KeyCode::Tab, &c);
            }
            for expected in (0..node.fields().len()).rev() {
                press(&mut p, KeyCode::BackTab, &c);
                let Screen::Edit { field, .. } = p.screen else {
                    panic!("edit closed")
                };
                assert_eq!(field, expected);
            }
        }
    }
}
pub struct Panel {
    pub screen: Screen,
    pub selected: usize,
    pub offset: usize,
    pub notice: String,
    follow_tail: bool,
    show_logs: bool,
    hits: Vec<(Rect, KeyCode)>,
    rows: Vec<(Rect, usize)>,
}
pub enum Intent {
    Keep,
    Close,
    Select(Option<String>),
    Save(Node),
    Remove(String),
    Request(Operation),
    ChooseJob(String),
    Reload,
    Plots,
    CopyLogs,
    OpenLog,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            screen: Screen::Nodes,
            selected: 0,
            offset: 0,
            notice: String::new(),
            follow_tail: true,
            show_logs: false,
            hits: Vec::new(),
            rows: Vec::new(),
        }
    }
}
fn text(value: &str) -> String {
    value
        .chars()
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .collect()
}
fn string(value: &Value, key: &str) -> String {
    text(value[key].as_str().unwrap_or("—"))
}
fn seconds(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v|v.is_finite()&&*v>=0.)
}
fn elapsed(value: &Value) -> String {
    let Some(value)=seconds(value) else{return "—".into()};
    let total=value.round() as u64;let (h,m,s)=(total/3600,total/60%60,total%60);
    if h>0&&s==0{format!("{h}h{m:02}m")}else if h>0{format!("{h}h{m:02}m{s:02}s")}else if m>0{format!("{m}m{s:02}s")}else{format!("{s}s")}
}
fn progress_stage(status:&Value)->String {
    let phase=status["phase"].as_str().unwrap_or("");let stage=status["stage"].as_str().unwrap_or(phase);
    let label=if status["state"]=="completed"||stage=="completed"{"Finished"}
        else if phase.contains("render")||stage=="finalize"{"Rendering images"}
        else if stage=="fetch"||phase.contains("fetch"){"Acquiring weather inputs"}
        else if stage=="prepare"||phase.starts_with("preparing:"){"Preparing forecast inputs"}
        else if stage=="initialize"||phase.contains("initializ"){"Initializing domains"}
        else if stage=="forecast"||phase=="integrating"||phase.contains("sync"){"Integrating forecast"}
        else if stage=="completed"{"Finished"}else{"Waiting for forecast progress"};
    format!("{} · {label}",status["state"].as_str().unwrap_or("refreshing").to_uppercase())
}
pub(crate) fn job_progress_text(status:&Value,compact:bool)->String {
    let mut lines=vec![progress_stage(status)];
    if let Some(error)=status["error"].as_str(){lines.push(format!("Needs attention: {}",text(error)));}
    let p=&status["progress"];
    let pipeline=&status["pipeline_progress"];
    let stage=pipeline["stage"].as_str().or_else(||status["stage"].as_str()).unwrap_or("");
    if seconds(&p["model_seconds"]).unwrap_or(0.)==0.&&matches!(stage,"fetch"|"prepare"|"initialize"){
        if pipeline["schema"]=="arwen.pipeline-progress.v1"{
            let phase=pipeline["phase"].as_str().unwrap_or(stage);
            lines.push(match phase{"cds_queued"=>"CDS is queuing the ERA5 request".into(),"cds_running"=>"CDS is preparing the ERA5 response".into(),"requesting"=>"Requesting ERA5 inputs".into(),"request_completed"=>"An ERA5 response has downloaded".into(),"validating"=>"Validating downloaded weather inputs".into(),"ready"=>"Weather inputs are ready".into(),other=>text(other)});
            lines.push(format!("Stage elapsed: {}",elapsed(&pipeline["wall_seconds"])));
            let acquisition=&pipeline["acquisition"];
            if let(Some(done),Some(total))=(acquisition["requests_completed"].as_u64(),acquisition["requests_total"].as_u64()){lines.push(format!("ERA5 requests: {done} / {total} complete"));}
            else if let(Some(done),Some(total))=(acquisition["files_completed"].as_u64(),acquisition["files_total"].as_u64()){lines.push(format!("Input files: {done} / {total} complete"));}
            if let Some(bytes)=acquisition["transferred_bytes"].as_u64(){lines.push(if acquisition["reused"]==true{"Using previously verified input files".into()}else{format!("Received {:.1} MiB",bytes as f64/1_048_576.)});}
            if let Some(times)=acquisition["forcing_times_total"].as_u64(){lines.push(format!("Boundary weather: {times} requested times over {} hours",acquisition["forcing_hours"].as_u64().map(|n|n.to_string()).unwrap_or_else(||"the requested".into())));}
            let prep=&pipeline["preparation"];
            if let(Some(index),Some(total))=(prep["phase_index"].as_u64().or_else(||prep["index"].as_u64()),prep["phases_total"].as_u64().or_else(||prep["count"].as_u64())){lines.push(format!("Preparation operation: {index} / {total}"));}
        }else{lines.push("Waiting for detailed input-preparation progress from the node.".into());}
        lines.push("Simulation steps and speed appear when integration starts.".into());
        return lines.join("\n");
    }
    if p["schema"]!="arwen.forecast-progress.v1"{
        if seconds(&status["model_elapsed_seconds"]).is_some(){lines.push(format!("Simulated time: {}",elapsed(&status["model_elapsed_seconds"])));}
        lines.push("Detailed timing is waiting for the next forecast progress report.".into());
        lines.push("G shows raw logs; Y copies them and O opens a text file.".into());
        return lines.join("\n");
    }
    let steps=p["outer_step"].as_u64().map(|v|v.to_string()).unwrap_or_else(||"—".into());
    lines.push(format!("Steps {steps} · Sim time {} / {}",elapsed(&p["model_seconds"]),elapsed(&p["run_seconds"])));
    let speed=seconds(&p["speed_x"]).filter(|s|*s>0.).map(|s|format!("{s:.1}× realtime")).unwrap_or_else(||"measuring".into());
    lines.push(format!("Wall time {} · Speed {speed}",elapsed(&p["wall_seconds"])));
    if let (Some(current),Some(total))=(seconds(&p["model_seconds"]),seconds(&p["run_seconds"]).filter(|v|*v>0.)){
        let ratio=(current/total).clamp(0.,1.);let width=if compact{12}else{24};let filled=(ratio*width as f64).round() as usize;
        lines.push(format!("Simulation [{}{}] {:.1}%", "#".repeat(filled),"-".repeat(width-filled),ratio*100.));
    }
    if let Some(domains)=p["domains"].as_array(){
        if compact{
            let upcoming=domains.iter().filter_map(|d|d["grid_id"].as_u64().map(|id|{
                let next=if seconds(&d["next_save_in_seconds"]).is_some(){format!("in {}",elapsed(&d["next_save_in_seconds"]))}else if seconds(&d["last_save_model_seconds"]).is_some(){format!("saved {}",elapsed(&d["last_save_model_seconds"]))}else{"pending".into()};
                format!("d{id:02} {next}")
            })).collect::<Vec<_>>().join("; ");
            lines.push(format!("Output: {upcoming}"));
        }else{
            for d in domains{if let Some(id)=d["grid_id"].as_u64(){
                let next=if seconds(&d["next_save_in_seconds"]).is_some(){format!("next in {}",elapsed(&d["next_save_in_seconds"]))}else{"no further save scheduled".into()};
                lines.push(format!("d{id:02} output every {} · {next} · last saved {}",elapsed(&d["history_interval_s"]),elapsed(&d["last_save_model_seconds"])));
            }}
        }
    }
    let checkpoint=&p["checkpoint"];
    lines.push(if seconds(&checkpoint["interval_seconds"])==Some(0.){"Checkpoints: disabled for this run".into()}
        else if seconds(&checkpoint["in_seconds"]).is_some(){format!("Next checkpoint in {} simulation time",elapsed(&checkpoint["in_seconds"]))}
        else if seconds(&checkpoint["last_saved_model_seconds"]).is_some(){format!("Checkpoint saved at {} simulated",elapsed(&checkpoint["last_saved_model_seconds"]))}
        else{"Checkpoint timing unavailable".into()});
    if let Some(valid)=p["valid_time"].as_str(){lines.push(format!("Forecast time: {} UTC",valid.trim_end_matches('Z').replace('T'," ")));}
    if status["state"]=="running"&&seconds(&p["model_seconds"]).zip(seconds(&p["run_seconds"])).is_some_and(|(now,end)|now>=end){lines.push("Simulation finished. Images are still being produced.".into());}
    lines.join("\n")
}
fn controls(
    buttons: &[(&str, KeyCode)],
    frame: &mut Frame,
    area: Rect,
    hits: &mut Vec<(Rect, KeyCode)>,
) {
    let mut x = area.x;
    let mut y = area.y;
    for (label, key) in buttons {
        let primary = *key == KeyCode::Enter;
        let label = format!("[{label}]");
        let width = (unicode_width::UnicodeWidthStr::width(label.as_str()) + 1)
            .min(area.width as usize) as u16;
        if x + width > area.right() {
            x = area.x;
            y += 1;
        }
        if y >= area.bottom() {
            break;
        }
        let rect = Rect::new(x, y, width, 1);
        let style = if label.contains("Stop") {
            Style::default().fg(theme::ERROR).bg(theme::ERROR_SURFACE)
        } else {
            theme::button(primary)
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(label, style),
                Span::styled(" ", Style::default().bg(theme::SURFACE)),
            ])),
            rect,
        );
        hits.push((rect, *key));
        x += width;
    }
}
fn contains(area: Rect, event: MouseEvent) -> bool {
    event.column >= area.x
        && event.column < area.right()
        && event.row >= area.y
        && event.row < area.bottom()
}
impl Panel {
    pub fn error(&mut self, message: String) {
        let previous = match &self.screen {
            Screen::Error { previous, .. } => previous.clone(),
            other => Box::new(other.clone()),
        };
        self.notice = "Esc returns to saved inputs. For an uncertain launch, refresh Jobs before starting another run.".into();
        self.screen = Screen::Error { message, previous };
        self.offset = 0;
    }
    pub fn open(&mut self, c: &Controller) {
        self.screen = Screen::Nodes;
        self.offset = 0;
        self.selected = c
            .store
            .active
            .as_ref()
            .and_then(|id| c.store.nodes.iter().position(|n| &n.id == id))
            .map(|i| i + 1)
            .unwrap_or(0);
        self.notice = if c.store.nodes.is_empty() {
            "No Linux nodes are saved. N adds a connection using your existing SSH access. L uses Local computer.".into()
        } else {
            "Enter uses the highlighted target. L uses Local computer. Delete reviews removal of a saved connection; remote jobs are unaffected.".into()
        };
    }
    pub fn update(&mut self, update: &Update) {
        match update {
            Update::Preview { operation, review } => {
                self.offset = 0;
                self.screen = Screen::Review {
                    operation: operation.clone(),
                    review: review.clone(),
                };
                self.notice = "Review the node and paths. Enter starts this exact request.".into();
            }
            Update::Started(id) => {
                self.offset = 0;
                self.follow_tail = true;
                self.show_logs = false;
                self.screen = Screen::Job;
                self.notice = format!(
                    "Started {id}. Closing this window leaves the job running on the node."
                );
            }
            Update::Stopped(id) => {
                self.screen = Screen::Job;
                self.notice = format!("Node confirmed job {id} has terminated.");
            }
            Update::Jobs => {
                self.screen = Screen::Jobs;
                self.selected = 0;
                self.notice = "Choose a job to reconnect and read its log.".into();
            }
            Update::Connected => {
                self.notice =
                    "Connected. This node can now be used to start and control ArWen jobs.".into()
            }
            Update::Failed(error) => self.error(error.clone()),
            Update::Status | Update::Logs | Update::PlanReviewed(_) | Update::ArtifactsSynced(_) | Update::ProcessedFrameSynced(_) | Update::ArtifactIndexed(_) => {}
        }
    }
    pub fn should_refresh(&self, c: &Controller) -> bool {
        self.should_refresh_connected(c,false)
    }
    pub fn should_refresh_connected(&self,c:&Controller,companion_open:bool)->bool {
        let Some(job) = c.store.selected().and_then(|node| node.last_job.as_deref()) else {
            return false;
        };
        if (!companion_open&&!matches!(self.screen, Screen::Job)) || c.pending.is_some() {
            return false;
        }
        if c.view.connection_error.is_none()&&c.view.runtime.is_some()&&c.view.status.as_ref().and_then(|s|s["id"].as_str())!=Some(job){
            return true;
        }
        let delay_ms = if c.view.connection_error.is_some() {
            15_000
        } else if c.view.logs_complete_for(job) {
            return false;
        } else if c.view.needs_log_drain(job) {
            200
        } else {
            1_000
        };
        c.view
            .last_job_refresh.or(c.view.last_refresh)
            .is_none_or(|at| at.elapsed().as_millis() >= delay_ms)
    }
    pub fn key(
        &mut self,
        key: KeyEvent,
        c: &Controller,
        products: Result<String, String>,
    ) -> Intent {
        let code = if matches!(self.screen, Screen::Edit { .. } | Screen::Resume { .. }) {
            key.code
        } else if let KeyCode::Char(ch) = key.code {
            KeyCode::Char(ch.to_ascii_lowercase())
        } else {
            key.code
        };
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let screen = self.screen.clone();
        if c.pending.is_some() && code == KeyCode::Esc {
            self.notice = "The node request continues. Remote jobs keep running; reconnect through Jobs before retrying a launch.".into();
            return Intent::Close;
        }
        if c.pending.is_some()
            && !matches!(
                code,
                KeyCode::Esc
                    | KeyCode::Up
                    | KeyCode::Down
                    | KeyCode::PageUp
                    | KeyCode::PageDown
                    | KeyCode::Home
                    | KeyCode::End
                    | KeyCode::Char('g')
            )
        {
            self.notice = "Waiting for the node request. Esc closes this panel; the request and remote job continue.".into();
            return Intent::Keep;
        }
        if code == KeyCode::Char('l') && matches!(screen, Screen::Nodes | Screen::Jobs | Screen::Job | Screen::Error { .. }) {
            self.screen = Screen::Nodes;
            self.selected = 0;
            self.notice = "Local computer selected. Any jobs already on a node keep running there.".into();
            return Intent::Select(None);
        }
        match screen {
            Screen::Error { previous, .. } => {
                if code == KeyCode::Esc {
                    self.screen = *previous;
                    self.offset = 0;
                } else {
                    self.scroll(code);
                }
                return Intent::Keep;
            }
            Screen::Edit {
                mut node,
                mut field,
                mut editing,
            } => {
                if code == KeyCode::Esc {
                    if editing {
                        editing = false;
                    } else {
                        self.open(c);
                        return Intent::Keep;
                    }
                } else if ctrl && matches!(code, KeyCode::Char('s' | 'S')) {
                    return Intent::Save(node);
                } else if code == KeyCode::Tab || code == KeyCode::Down && !editing {
                    field = (field + 1) % node.fields().len();
                    editing = false;
                } else if code == KeyCode::BackTab || code == KeyCode::Up && !editing {
                    field = (field + node.fields().len() - 1) % node.fields().len();
                    editing = false;
                } else if code == KeyCode::Enter {
                    editing = !editing;
                } else if editing {
                    let mut value = node.fields()[field].1.to_string();
                    if ctrl && code == KeyCode::Char('u') {
                        value.clear();
                    } else if code == KeyCode::Backspace {
                        value.pop();
                    } else if let KeyCode::Char(ch) = code {
                        if !ctrl && !ch.is_control() && value.len() < 4096 {
                            value.push(ch);
                        }
                    }
                    node.set_field(field, value);
                }
                self.screen = Screen::Edit {
                    node,
                    field,
                    editing,
                };
                return Intent::Keep;
            }
            Screen::Resume {
                job,
                mut checkpoint,
                mut output,
                mut field,
            } => {
                if code == KeyCode::Esc {
                    self.screen = Screen::Job;
                    return Intent::Keep;
                }
                if code == KeyCode::Enter {
                    return Intent::Request(Operation::Resume {
                        job,
                        checkpoint,
                        output,
                        preview: true,
                        binding: None,
                    });
                }
                if matches!(
                    code,
                    KeyCode::Tab | KeyCode::BackTab | KeyCode::Up | KeyCode::Down
                ) {
                    field = 1 - field;
                } else {
                    let value = if field == 0 {
                        &mut checkpoint
                    } else {
                        &mut output
                    };
                    if ctrl && code == KeyCode::Char('u') {
                        value.clear();
                    } else if code == KeyCode::Backspace {
                        value.pop();
                    } else if let KeyCode::Char(ch) = code {
                        if !ctrl && !ch.is_control() && value.len() < 4096 {
                            value.push(ch);
                        }
                    }
                }
                self.screen = Screen::Resume {
                    job,
                    checkpoint,
                    output,
                    field,
                };
                return Intent::Keep;
            }
            Screen::Review { operation, review } => {
                if code == KeyCode::Esc {
                    self.screen = Screen::Nodes;
                    self.notice =
                        "Launch review closed. Jobs shows requests already sent to the node."
                            .into();
                } else if code == KeyCode::Enter && key.kind == KeyEventKind::Press {
                    if matches!(operation,Operation::StartPlan{..}){return Intent::Request(operation);}
                    match operation.confirmed(&review) {
                        Ok(op) => return Intent::Request(op),
                        Err(error) => self.notice = error,
                    }
                } else {
                    self.scroll(code);
                }
                return Intent::Keep;
            }
            Screen::Stop { job } => {
                if code == KeyCode::Esc {
                    self.screen = Screen::Job;
                } else if code == KeyCode::Enter && key.kind == KeyEventKind::Press {
                    return Intent::Request(Operation::Stop { job });
                }
                return Intent::Keep;
            }
            Screen::Remove(node) => {
                if code == KeyCode::Esc {
                    self.open(c);
                } else if code == KeyCode::Enter && key.kind == KeyEventKind::Press {
                    return Intent::Remove(node.id);
                } else {
                    self.scroll(code);
                }
                return Intent::Keep;
            }
            Screen::Nodes => match code {
                KeyCode::Esc => return Intent::Close,
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => self.selected = (self.selected + 1).min(c.store.nodes.len()),
                KeyCode::Enter | KeyCode::Char('u') => {
                    return Intent::Select(
                        self.selected
                            .checked_sub(1)
                            .and_then(|i| c.store.nodes.get(i))
                            .map(|n| n.id.clone()),
                    )
                }
                KeyCode::Char('n') => {
                    self.screen = Screen::Edit {
                        node: Node::blank(),
                        field: 0,
                        editing: false,
                    };
                    self.notice = "SSH uses existing keys/agent and verified known hosts. No passwords or key contents are saved.".into();
                }
                KeyCode::Char('e') => {
                    if let Some(node) = self
                        .selected
                        .checked_sub(1)
                        .and_then(|i| c.store.nodes.get(i))
                    {
                        self.screen = Screen::Edit {
                            node: node.clone(),
                            field: 0,
                            editing: false,
                        };
                    }
                }
                KeyCode::Delete => {
                    if let Some(node) = self.selected.checked_sub(1).and_then(|i| c.store.nodes.get(i)) {
                        self.screen = Screen::Remove(node.clone());
                        self.offset = 0;
                        self.notice = "Review this saved connection. Removing it sends no request to the node and stops no jobs.".into();
                    } else {
                        self.notice = "Local computer cannot be removed. Highlight a saved Linux node to remove its connection.".into();
                    }
                }
                KeyCode::Char('r') => return Intent::Reload,
                _ => {}
            },
            Screen::Jobs => match code {
                KeyCode::Esc => {
                    self.open(c);
                    return Intent::Keep;
                }
                KeyCode::Up => self.selected = self.selected.saturating_sub(1),
                KeyCode::Down => {
                    self.selected = (self.selected + 1).min(c.view.jobs.len().saturating_sub(1))
                }
                KeyCode::Enter | KeyCode::Char('v') => {
                    if let Some(job) = c
                        .view
                        .jobs
                        .get(self.selected)
                        .and_then(|v| v["id"].as_str())
                    {
                        self.screen = Screen::Job;
                        self.offset = 0;
                        self.follow_tail = true;
                        self.show_logs = false;
                        return Intent::ChooseJob(job.into());
                    } else if c.view.jobs.is_empty() && key.kind == KeyEventKind::Press {
                        return self.start_review(c, products);
                    }
                }
                KeyCode::Char('r') => return Intent::Request(Operation::List),
                _ => {}
            },
            Screen::Job => {
                if code == KeyCode::Esc {
                    self.screen = Screen::Jobs;
                    return Intent::Keep;
                }
                let job = c.store.selected().and_then(|n| n.last_job.clone());
                match code {
                    KeyCode::Char('g') => {self.show_logs=!self.show_logs;self.offset=0;}
                    KeyCode::Char('y' | 'Y') => return Intent::CopyLogs,
                    KeyCode::Char('o' | 'O') => return Intent::OpenLog,
                    KeyCode::Char('r') => {
                        if let Some(job) = job {
                            return Intent::Request(Operation::Logs {
                                job,
                                cursor: c.view.cursor,
                            });
                        }
                    }
                    KeyCode::Char('x') => {
                        if let Some(job) = job {
                            self.screen = Screen::Stop { job };
                        }
                    }
                    KeyCode::Char('c') => {
                        if let Some(job) = job {
                            self.screen = Screen::Resume {
                                job,
                                checkpoint: "latest".into(),
                                output: String::new(),
                                field: 0,
                            };
                        }
                    }
                    _ => self.scroll(code),
                }
            }
        }
        if matches!(self.screen, Screen::Nodes | Screen::Jobs | Screen::Job) {
            if c.store.selected().is_none() && matches!(code, KeyCode::Char('p' | 'j' | 's') | KeyCode::F(7)) {
                self.notice = if c.store.nodes.is_empty() { "No Linux node is saved. Press N to add one, or L to use Local computer." } else { "Select a saved Linux node with Enter before connecting or reviewing a start. L uses Local computer." }.into();
                return Intent::Keep;
            }
            match code {
                KeyCode::Char('p') => return Intent::Request(Operation::Probe),
                KeyCode::Char('j') => return Intent::Request(Operation::List),
                KeyCode::Char('b') => return Intent::Plots,
                KeyCode::Char('s') | KeyCode::F(7) => return self.start_review(c, products),
                _ => {}
            }
        }
        Intent::Keep
    }
    fn start_review(&mut self, c: &Controller, products: Result<String, String>) -> Intent {
        if c.store.selected().is_none() {
            self.notice = "Select a saved Linux node in Nodes before reviewing a start. L uses Local computer.".into();
            return Intent::Keep;
        }
        match products {
            Ok(products) => Intent::Request(Operation::Start { products, preview: true, binding: None }),
            Err(error) => { self.notice = error; Intent::Keep }
        }
    }
    fn scroll(&mut self, code: KeyCode) {
        if matches!(self.screen, Screen::Job) {
            match code {
                KeyCode::Up | KeyCode::PageUp | KeyCode::Home => self.follow_tail = false,
                KeyCode::End => self.follow_tail = true,
                _ => {}
            }
        }
        match code {
            KeyCode::Up => self.offset = self.offset.saturating_sub(1),
            KeyCode::Down => self.offset = self.offset.saturating_add(1),
            KeyCode::PageUp => self.offset = self.offset.saturating_sub(8),
            KeyCode::PageDown => self.offset = self.offset.saturating_add(8),
            KeyCode::Home => self.offset = 0,
            KeyCode::End => self.offset = usize::MAX / 2,
            _ => {}
        }
    }
    pub fn paste(&mut self, value: &str) {
        let value: String = value
            .trim_end_matches(['\r', '\n'])
            .chars()
            .filter(|c| !c.is_control())
            .take(4096)
            .collect();
        match &mut self.screen {
            Screen::Edit {
                node,
                field,
                editing: true,
            } => {
                let mut current = node.fields()[*field].1.to_string();
                if current.len() + value.len() <= 4096 {
                    current.push_str(&value);
                    node.set_field(*field, current);
                }
            }
            Screen::Resume {
                checkpoint,
                output,
                field,
                ..
            } => {
                let target = if *field == 0 { checkpoint } else { output };
                if target.len() + value.len() <= 4096 {
                    target.push_str(&value);
                }
            }
            _ => {}
        }
    }
    pub fn mouse(
        &mut self,
        event: MouseEvent,
        c: &Controller,
        products: Result<String, String>,
    ) -> Intent {
        if matches!(
            event.kind,
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown
        ) {
            return self.key(
                KeyEvent::new(
                    if event.kind == MouseEventKind::ScrollUp {
                        KeyCode::Up
                    } else {
                        KeyCode::Down
                    },
                    KeyModifiers::NONE,
                ),
                c,
                products,
            );
        }
        if event.kind != MouseEventKind::Down(MouseButton::Left) {
            return Intent::Keep;
        }
        if let Some((_, key)) = self.hits.iter().find(|(area, _)| contains(*area, event)) {
            let key = *key;
            return self.key(
                KeyEvent::new(
                    key,
                    if key == KeyCode::Char('S') {
                        KeyModifiers::CONTROL
                    } else {
                        KeyModifiers::NONE
                    },
                ),
                c,
                products,
            );
        }
        if let Some((_, index)) = self.rows.iter().find(|(area, _)| contains(*area, event)) {
            match &mut self.screen {
                Screen::Edit { field, editing, .. } => {
                    *field = *index;
                    *editing = true;
                }
                Screen::Resume { field, .. } => *field = (*index).min(1),
                Screen::Nodes => self.selected = (*index).min(c.store.nodes.len()),
                _ => self.selected = *index,
            }
        }
        Intent::Keep
    }
    pub fn draw(&mut self, frame: &mut Frame, area: Rect, c: &Controller, plots: &str) {
        self.hits.clear();
        self.rows.clear();
        frame.render_widget(Clear, area);
        let target = c
            .store
            .selected()
            .map(|n| format!("{} · {}", n.name, n.host))
            .unwrap_or_else(|| "Local computer".into());
        let title = format!(" Nodes — target: {} ", text(&target));
        let block = theme::panel(&title).border_style(Style::default().fg(
            if matches!(self.screen, Screen::Error { .. }) {
                theme::ERROR
            } else {
                theme::SKY
            },
        ));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let rows = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(2),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(inner);
        let connection = if let Some(pending) = &c.pending {
            if matches!(pending.operation,Operation::Probe)&&c.view.runtime.is_none(){"Connecting to the selected node…".into()}
            else{format!("Connected · {}…",match pending.operation{Operation::ArtifactIndex{..}=>"checking saved forecast times",Operation::SyncArtifacts{..}=>"loading saved forecast output",Operation::Logs{..}|Operation::Status{..}=>"updating forecast progress",_=>"processing the requested action"})}
        } else if let Some(error) = c.load_error.as_ref().or(c.view.connection_error.as_ref()) {
            format!("Needs attention: {error}")
        } else if c.store.nodes.is_empty() {
            "Local computer is selected. Add a Linux node to use existing SSH access.".into()
        } else if c.store.selected().is_none() {
            "Local computer is selected. Use a saved node to connect or review a remote start.".into()
        } else {
            format!("Plots: {plots}. Jobs continue on the node when this TUI closes.")
        };
        frame.render_widget(
            Paragraph::new(text(&connection))
                .wrap(Wrap { trim: false })
                .style(
                    if c.load_error
                        .as_ref()
                        .or(c.view.connection_error.as_ref())
                        .is_some()
                    {
                        theme::notice(true)
                    } else if c.pending.is_some() {
                        theme::notice(false)
                    } else {
                        Style::default().fg(theme::MUTED)
                    },
                ),
            rows[0],
        );
        let mut buttons = Vec::new();
        match &self.screen {
            Screen::Error { message, .. } => {
                let message = format!("Node request needs attention\n\n{message}");
                self.paragraph(frame, rows[1], &message);
                buttons.extend([("PgDn More", KeyCode::PageDown), ("L Local computer", KeyCode::Char('l')), ("Esc Back", KeyCode::Esc)]);
            }
            Screen::Nodes => {
                let mut choices = vec![format!(
                    "{}Local computer",
                    if c.store.active.is_none() {
                        "● "
                    } else {
                        "  "
                    }
                )];
                choices.extend(c.store.nodes.iter().map(|n| {
                    format!(
                        "{}{}  {}",
                        if c.store.active.as_ref() == Some(&n.id) {
                            "● "
                        } else {
                            "  "
                        },
                        text(&n.name),
                        text(&n.host)
                    )
                }));
                if let Some(runtime) = &c.view.runtime {
                    choices.push(format!(
                        "Connected ArWen {} · {}",
                        string(&runtime["runtime"], "version"),
                        string(&runtime["runtime"], "python")
                    ));
                }
                self.list(frame, rows[1], choices);
                buttons.extend([
                    ("Enter Use", KeyCode::Enter),
                    ("L Local computer", KeyCode::Char('l')),
                    ("N Add", KeyCode::Char('n')),
                    ("E Edit", KeyCode::Char('e')),
                    ("Delete Remove", KeyCode::Delete),
                    ("P Connect", KeyCode::Char('p')),
                    ("J Jobs", KeyCode::Char('j')),
                    ("S Review start", KeyCode::Char('s')),
                    ("B Plots", KeyCode::Char('b')),
                    ("R Reload", KeyCode::Char('r')),
                    ("Esc Close", KeyCode::Esc),
                ]);
            }
            Screen::Remove(node) => {
                let content = format!("Remove saved node {}?\nHost: {}\n\nRemote jobs keep running; remote files are unchanged.\nOnly this saved connection and its last-job link are removed.\nRe-add the same host and workspace to reconnect.\n\nWorkspace: {}\nLast linked job: {}", text(&node.name), text(&node.host), text(&node.workspace), node.last_job.as_deref().unwrap_or("none"));
                self.paragraph(frame, rows[1], &content);
                buttons.extend([("Enter Remove profile", KeyCode::Enter), ("PgDn More", KeyCode::PageDown), ("Esc Cancel", KeyCode::Esc)]);
            }
            Screen::Edit {
                node,
                field,
                editing,
            } => {
                let mut choices: Vec<_> = node
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(i, (label, value))| {
                        let label = if i == 5 {
                            "Output parent on node (optional)"
                        } else {
                            label
                        };
                        format!(
                            "{label}: {}{}",
                            if value.is_empty() { "(empty)" } else { value },
                            if i == *field && *editing { " ▏" } else { "" }
                        )
                    })
                    .collect();
                choices[5].push_str("  [blank = workspace]");
                self.selected = *field;
                self.list(frame, rows[1], choices);
                buttons.extend([
                    ("Enter Edit/Done", KeyCode::Enter),
                    ("Tab Next", KeyCode::Tab),
                    ("Ctrl+S Save", KeyCode::Char('S')),
                    ("Esc Back", KeyCode::Esc),
                ]);
            }
            Screen::Jobs => {
                if c.view.jobs.is_empty() {
                    self.paragraph(frame, rows[1], "No jobs are listed for this node's workspace.\n\nEnter or S reviews your first start using the saved configuration. Nothing starts until you confirm that review. R refreshes the job list.");
                    buttons.push(("Enter Review first start", KeyCode::Enter));
                } else {
                    let choices = c
                    .view
                    .jobs
                    .iter()
                    .map(|v| format!("{:12} {}", string(v, "state"), string(v, "id")))
                    .collect();
                    self.list(frame, rows[1], choices);
                    buttons.push(("Enter View", KeyCode::Enter));
                }
                buttons.extend([
                    ("R Refresh", KeyCode::Char('r')),
                    ("S Review start", KeyCode::Char('s')),
                    ("L Local computer", KeyCode::Char('l')),
                    ("Esc Nodes", KeyCode::Esc),
                ]);
            }
            Screen::Job => {
                let status = c.view.status.as_ref();
                let job = c
                    .store
                    .selected()
                    .and_then(|n| n.last_job.as_deref())
                    .unwrap_or("No job selected");
                let mut lines = if self.show_logs {format!(
                    "Job: {job}\nLast known state: {}\nOutput: {}\n\n{}",
                    status
                        .map(|v| string(v, "state"))
                        .unwrap_or_else(|| "refreshing".into()),
                    status
                        .map(|v| string(v, "outdir"))
                        .unwrap_or_else(|| "—".into()),
                    c.view.log
                )}else{job_progress_text(status.unwrap_or(&Value::Null),rows[1].width<90)};
                if self.show_logs&&c.view.log.is_empty() {
                    lines.push_str("\nNo log text received yet. R refreshes.");
                }
                self.paragraph(frame, rows[1], &lines);
                buttons.extend([
                    (if self.show_logs{"G Progress"}else{"G Raw logs"},KeyCode::Char('g')),
                    ("Y Copy logs", KeyCode::Char('y')),
                    ("O Open log", KeyCode::Char('o')),
                    ("R Refresh", KeyCode::Char('r')),
                    ("X Stop", KeyCode::Char('x')),
                    ("C Resume", KeyCode::Char('c')),
                    ("PgUp/PgDn", KeyCode::PageDown),
                    ("L Local computer", KeyCode::Char('l')),
                    ("Esc Jobs", KeyCode::Esc),
                ]);
            }
            Screen::Review { operation, review } => {
                let content = format!("{} ON {}\n\nRemote configuration: {}\nNew output folder: {}\nPlots: {}\nCheckpoint: {}\n\nExact launch and input hashes:\n{}", operation.action().to_uppercase(), target, string(review, "config"), string(review, "outdir"), string(review, "products"), string(review, "checkpoint"), serde_json::to_string_pretty(review).unwrap_or_default());
                self.paragraph(frame, rows[1], &content);
                buttons.extend([
                    ("Enter Start on node", KeyCode::Enter),
                    ("PgDn More", KeyCode::PageDown),
                    ("Esc Cancel", KeyCode::Esc),
                ]);
            }
            Screen::Stop { job } => {
                frame.render_widget(Paragraph::new(format!("Stop job {job}\non {target}?\n\nThis sends termination to this job's verified processes. Saved output remains. The TUI reports stopped only after the node confirms termination.")).wrap(Wrap { trim: false }), rows[1]);
                buttons.extend([
                    ("Enter Stop job", KeyCode::Enter),
                    ("Esc Cancel", KeyCode::Esc),
                ]);
            }
            Screen::Resume {
                job,
                checkpoint,
                output,
                field,
            } => {
                let value = if *field == 0 { checkpoint } else { output };
                let suffix: String = value
                    .chars()
                    .rev()
                    .take(rows[1].width as usize * 2 - 2)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                let detail = format!(
                    "Source job: {job}\nEditing {} · Ctrl+U clears\n{}{suffix}▏",
                    if *field == 0 {
                        "checkpoint"
                    } else {
                        "new output"
                    },
                    if suffix.len() < value.len() {
                        "…"
                    } else {
                        ""
                    }
                );
                let field = *field;
                let choices = vec![
                    format!("Checkpoint (latest or absolute path): {checkpoint}"),
                    format!("New output folder (blank = unique): {output}"),
                ];
                let parts =
                    Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(rows[1]);
                self.selected = field;
                self.list(frame, parts[0], choices);
                frame.render_widget(
                    Paragraph::new(text(&detail)).wrap(Wrap { trim: false }),
                    parts[1],
                );
                buttons.extend([
                    ("Enter Review", KeyCode::Enter),
                    ("Tab Next", KeyCode::Tab),
                    ("Esc Cancel", KeyCode::Esc),
                ]);
            }
        }
        let notice = if let Screen::Edit {
            node,
            field,
            editing: true,
        } = &self.screen
        {
            let (label, value) = node.fields()[*field];
            let suffix: String = value
                .chars()
                .rev()
                .take(rows[2].width as usize * 2 - 2)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            format!(
                "{label} · Ctrl+U clears · Enter done\n{}{suffix}▏",
                if suffix.len() < value.len() {
                    "…"
                } else {
                    ""
                }
            )
        } else if matches!(self.screen, Screen::Job)&&!self.show_logs {
            "Output and checkpoint countdowns use simulation time. Wall time and speed measure the forecast integration.".into()
        } else if matches!(self.screen, Screen::Jobs) && c.view.jobs.is_empty() {
            "The job list is empty. Enter/S requests a start review; the later confirmation is what starts a new job.".into()
        } else {
            self.notice.clone()
        };
        frame.render_widget(
            Paragraph::new(text(&notice))
                .style(Style::default().fg(theme::MUTED))
                .wrap(Wrap { trim: false }),
            rows[2],
        );
        controls(&buttons, frame, rows[3], &mut self.hits);
    }
    fn list(&mut self, frame: &mut Frame, area: Rect, choices: Vec<String>) {
        let length = choices.len();
        let mut state = ListState::default().with_selected(Some(self.selected));
        frame.render_stateful_widget(
            List::new(choices)
                .highlight_symbol("> ")
                .highlight_style(theme::selected()),
            area,
            &mut state,
        );
        for index in state.offset()..length {
            let y = area.y + (index - state.offset()) as u16;
            if y < area.bottom() {
                self.rows.push((Rect::new(area.x, y, area.width, 1), index));
            }
        }
    }
    fn paragraph(&mut self, frame: &mut Frame, area: Rect, value: &str) {
        let value = text(value);
        let lines = crate::log_display_rows(&value, area.width as usize);
        let last = lines.len().saturating_sub(area.height as usize);
        self.offset = if matches!(self.screen, Screen::Job) && self.show_logs && self.follow_tail {
            last
        } else {
            self.offset.min(last)
        };
        frame.render_widget(
            Paragraph::new(
                lines
                    .into_iter()
                    .skip(self.offset)
                    .map(Line::raw)
                    .collect::<Vec<_>>(),
            ),
            area,
        );
    }
}
