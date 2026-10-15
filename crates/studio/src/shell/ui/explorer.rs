//! System Explorer (GEN-57): the left navigation tree, modeled on Zemax
//! OpticStudio's System Explorer — grouped sections whose items open the
//! matching tab in the center workspace.

use dioxus::prelude::*;

use super::studio::{Shell, WorkspaceTab};

/// One tree entry: label + the workspace tab it opens.
type Item = (&'static str, WorkspaceTab);

/// The explorer tree. Clicking an item switches the center workspace tab;
/// the active tab's items are highlighted.
#[component]
pub fn SystemExplorer() -> Element {
    let studio = use_context::<Shell>();
    let active = studio.tab();
    rsx! {
        div { class: "panel tree",
            h2 { "System Explorer" }
            {section(studio, active, "Prescription", &[
                ("Lens Data Editor", WorkspaceTab::LensData),
                ("Prescription Text (.zmx)", WorkspaceTab::Text),
            ])}
            {section(studio, active, "System", &[
                ("Wavelengths", WorkspaceTab::Setup),
                ("Source", WorkspaceTab::Setup),
                ("Propagation Engine", WorkspaceTab::Setup),
                ("Paraxial Data", WorkspaceTab::Setup),
            ])}
            {section(studio, active, "Viewers", &[
                ("Layout — Strip + PSF", WorkspaceTab::Layout),
                ("PSF / Metrics Analysis", WorkspaceTab::Analysis),
            ])}
            {section(studio, active, "Inverse Design", &[
                ("Carrier Sliders + Run Console", WorkspaceTab::Run),
            ])}
        }
    }
}

fn section(studio: Shell, active: WorkspaceTab, title: &'static str, items: &[Item]) -> Element {
    let items = items.to_vec();
    rsx! {
        div { class: "tree-group",
            div { class: "tree-h", "{title}" }
            for (label, tab) in items {
                button {
                    key: "{label}",
                    class: if tab == active { "tree-item active" } else { "tree-item" },
                    onclick: {
                        move |_| {
                            let mut s = studio;
                            s.set_tab(tab);
                        }
                    },
                    "{label}"
                }
            }
        }
    }
}
