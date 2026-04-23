//! ITfEditSession — applies engine actions to the TSF context.
//!
//! Pattern from karukan-tsf (MIT/Apache-2.0): composition lifetime lives in a
//! shared `Rc<RefCell<Option<ITfComposition>>>` so that StartComposition /
//! EndComposition during DoEditSession propagates back to the service after
//! the synchronous call returns.

use std::cell::RefCell;
use std::rc::Rc;

use windows::core::*;
use windows::Win32::Foundation::FALSE;
use windows::Win32::UI::TextServices::*;

use crate::engine_bridge::Action;

#[implement(ITfEditSession)]
pub struct ActionEditSession {
    context: ITfContext,
    actions: Vec<Action>,
    composition: Rc<RefCell<Option<ITfComposition>>>,
    composition_sink: ITfCompositionSink,
    atom_input: u32,
    #[allow(dead_code)]
    atom_converted: u32,
}

impl ActionEditSession {
    pub fn new(
        context: ITfContext,
        actions: Vec<Action>,
        composition: Rc<RefCell<Option<ITfComposition>>>,
        composition_sink: ITfCompositionSink,
        atom_input: u32,
        atom_converted: u32,
    ) -> Self {
        Self {
            context,
            actions,
            composition,
            composition_sink,
            atom_input,
            atom_converted,
        }
    }
}

impl ITfEditSession_Impl for ActionEditSession_Impl {
    fn DoEditSession(&self, ec: u32) -> Result<()> {
        for action in &self.actions {
            match action {
                Action::UpdatePreedit { text } => self.update_preedit(ec, text)?,
                Action::Commit { text } => self.commit_text(ec, text)?,
                Action::EndComposition => self.discard_composition(ec)?,
                // Candidate window show/hide happen via a separate side-channel
                // in key_event_sink after the edit session returns; no-op here.
                Action::ShowCandidates { .. } | Action::HideCandidates => {}
            }
        }
        Ok(())
    }
}

impl ActionEditSession {
    fn ensure_composition(&self, ec: u32) -> Result<()> {
        let mut comp = self.composition.borrow_mut();
        if comp.is_some() {
            return Ok(());
        }
        unsafe {
            let insert_at_sel: ITfInsertAtSelection = self.context.cast()?;
            let range = insert_at_sel.InsertTextAtSelection(ec, TF_IAS_QUERYONLY, &[])?;
            let ctx_comp: ITfContextComposition = self.context.cast()?;
            let new_comp = ctx_comp.StartComposition(ec, &range, &self.composition_sink)?;
            *comp = Some(new_comp);
        }
        Ok(())
    }

    fn end_composition(&self, ec: u32) -> Result<()> {
        let mut comp = self.composition.borrow_mut();
        if let Some(composition) = comp.take() {
            unsafe {
                composition.EndComposition(ec)?;
            }
        }
        Ok(())
    }

    /// End the composition after clearing any leftover preedit text. Used when
    /// the composition should vanish without committing (e.g. Escape, or when
    /// Backspace reduces the preedit to empty). `EndComposition` only
    /// finalizes — it does not erase — so the SetText must happen first.
    fn discard_composition(&self, ec: u32) -> Result<()> {
        {
            let comp = self.composition.borrow();
            if let Some(ref composition) = *comp {
                unsafe {
                    if let Ok(range) = composition.GetRange() {
                        let _ = range.SetText(ec, 0, &[]);
                    }
                }
            }
        }
        self.end_composition(ec)
    }

    fn update_preedit(&self, ec: u32, text: &str) -> Result<()> {
        if text.is_empty() {
            return self.discard_composition(ec);
        }
        self.ensure_composition(ec)?;

        let comp = self.composition.borrow();
        if let Some(ref composition) = *comp {
            unsafe {
                let range = composition.GetRange()?;
                let wide: Vec<u16> = text.encode_utf16().collect();
                range.SetText(ec, 0, &wide)?;

                // Apply display attribute (dotted underline) across the full
                // composition range so the user can see what is still in IME
                // composition vs already-committed document text. Non-fatal
                // if the property lookup fails — the composition just goes
                // un-styled.
                if self.atom_input != 0 {
                    let _ = self.apply_display_attribute(ec, &range, self.atom_input);
                }

                // Put the caret at the end of the composition.
                let caret_range = range.Clone()?;
                caret_range.Collapse(ec, TF_ANCHOR_END)?;
                let sel = TF_SELECTION {
                    range: core::mem::ManuallyDrop::new(Some(caret_range)),
                    style: TF_SELECTIONSTYLE {
                        ase: TF_AE_END,
                        fInterimChar: FALSE,
                    },
                };
                self.context.SetSelection(ec, &[sel])?;
            }
        }
        Ok(())
    }

    unsafe fn apply_display_attribute(&self, ec: u32, range: &ITfRange, atom: u32) -> Result<()> {
        let prop: ITfProperty = unsafe { self.context.GetProperty(&GUID_PROP_ATTRIBUTE)? };
        let variant = windows_core::VARIANT::from(atom as i32);
        unsafe { prop.SetValue(ec, range, &variant)? };
        Ok(())
    }

    fn commit_text(&self, ec: u32, text: &str) -> Result<()> {
        // Unified path: make sure a composition exists (creating a zero-length
        // one at the current selection if not), replace its content with
        // `text`, move the caret to the end, then end the composition. Both
        // regular conversion commits AND Empty-phase punctuation commits go
        // through this, so the two can never diverge.
        //
        // The earlier `InsertTextAtSelection(TF_IAS_NOQUERY)` shortcut for
        // the no-composition case was crashing the host: some implementations
        // return S_OK with `ppRange = NULL`, which windows-rs wraps as an
        // ITfRange that calls Release on a null COM pointer at drop.
        self.ensure_composition(ec)?;

        // Scope the immutable borrow so `end_composition`'s `borrow_mut`
        // below doesn't panic with BorrowMutError (that propagated through
        // the FFI boundary and aborted Notepad).
        {
            let comp = self.composition.borrow();
            if let Some(ref composition) = *comp {
                unsafe {
                    let range = composition.GetRange()?;
                    let wide: Vec<u16> = text.encode_utf16().collect();
                    range.SetText(ec, 0, &wide)?;

                    let caret = range.Clone()?;
                    caret.Collapse(ec, TF_ANCHOR_END)?;
                    let sel = TF_SELECTION {
                        range: core::mem::ManuallyDrop::new(Some(caret)),
                        style: TF_SELECTIONSTYLE {
                            ase: TF_AE_END,
                            fInterimChar: FALSE,
                        },
                    };
                    self.context.SetSelection(ec, &[sel])?;
                }
            }
        }
        self.end_composition(ec)
    }
}
