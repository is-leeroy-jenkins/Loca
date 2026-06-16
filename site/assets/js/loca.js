
/*
 * ==========================================================================================
 *  Loca Documentation JavaScript
 *  File: docs/assets/js/loca.js
 *
 *  Purpose:
 *      Provides safe progressive enhancements for the Loca MkDocs Material documentation
 *      site. The script improves API reference navigation, long-code readability, table
 *      inspection, heading linking, page utility actions, and reading progress without
 *      requiring external libraries.
 *
 *  Features:
 *      - API tools panel with search, expand all, collapse all, and clear filter
 *      - Scroll-to-top control
 *      - Reading progress bar
 *      - Heading anchor copy buttons
 *      - Page copy and print tools
 *      - Large-table filtering
 *      - Code-block language labels
 *      - Long-code expand/collapse controls
 *      - External-link hardening
 *      - Search placeholder customization
 *      - Active table-of-contents highlighting
 *      - Navigation scroll memory
 *      - Page path metadata
 *      - Keyboard focus mode
 *      - mkdocstrings API member badges
 *      - Mermaid block detection guard
 *
 *  Compatibility:
 *      - MkDocs Material
 *      - mkdocstrings
 *      - Modern Chromium, Edge, Firefox, Safari
 *
 *  Notes:
 *      This file intentionally avoids external dependencies, network calls, analytics,
 *      cookies, and storage of user-authored page content.
 * ==========================================================================================
 */

( function()
{
	"use strict";
	const LocaDocs = {
		config: {
			appName: "Loca",
			scrollTopId: "loca-scroll-top",
			pageToolsId: "loca-page-tools",
			progressId: "loca-reading-progress",
			apiSearchId: "loca-api-search",
			initializedAttribute: "data-loca-enhanced",
			navScrollKey: "loca-docs-nav-scroll",
			contentSelector: ".md-content__inner",
			headingSelector: ".md-typeset h2[id], .md-typeset h3[id], .md-typeset h4[id]",
			navSelector: ".md-nav--primary .md-nav__list",
			tocSelector: ".md-nav--secondary",
			tableSelector: ".md-typeset table:not([data-loca-no-filter])",
			codeSelector: ".md-typeset pre > code",
			apiObjectSelector:
					".doc.doc-object, .doc-class, .doc-function, .doc-method, .doc-attribute, .doc-property",
			apiToolsClass: "loca-api-tools",
			apiHiddenClass: "loca-api-hidden",
			apiBadgeClass: "loca-api-badge",
			tableToolsClass: "loca-table-tools",
			tableFilterClass: "loca-table-filter",
			tableCountClass: "loca-table-count",
			headingLinkClass: "loca-heading-link",
			codeLabelClass: "loca-code-label",
			codeToggleClass: "loca-code-toggle",
			codeCollapsedClass: "loca-code-collapsed",
			externalLinkClass: "loca-external-link",
			tocActiveClass: "loca-toc-active",
			keyboardModeClass: "loca-keyboard-mode",
			maxCollapsedCodeHeight: 420,
			largeTableMinimumRows: 8,
			scrollTopVisibleAt: 420
		},
		state: {
			pageReady: false,
			scrollTicking: false,
			resizeTicking: false
		},
		init: function()
		{
			if( document.documentElement.getAttribute(
					this.config.initializedAttribute ) === "true" )
			{
				return;
			}
			document.documentElement.setAttribute( this.config.initializedAttribute,
					"true" );
			this.enhanceExternalLinks();
			this.customizeSearch();
			this.addReadingProgress();
			this.addScrollTopButton();
			this.addPagePathMetadata();
			this.addPageTools();
			this.addHeadingLinks();
			this.addTableFilters();
			this.addCodeLabels();
			this.addCodeToggles();
			this.enhanceApiReference();
			this.addApiTools();
			this.enhanceTocProgress();
			this.restoreNavigationScroll();
			this.enhanceKeyboardFocus();
			this.addMermaidGuard();
			this.bindLifecycleEvents();
			this.state.pageReady = true;
			this.updateReadingProgress();
			this.updateScrollTopVisibility();
			this.updateTocProgress();
		},
		bindLifecycleEvents: function()
		{
			const self = this;
			window.addEventListener( "scroll", function()
			{
				if( !self.state.scrollTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateScrollTopVisibility();
						self.updateTocProgress();
						self.state.scrollTicking = false;
					} );
					self.state.scrollTicking = true;
				}
			}, { passive: true } );
			window.addEventListener( "resize", function()
			{
				if( !self.state.resizeTicking )
				{
					window.requestAnimationFrame( function()
					{
						self.updateReadingProgress();
						self.updateTocProgress();
						self.state.resizeTicking = false;
					} );
					self.state.resizeTicking = true;
				}
			}, { passive: true } );
			document.addEventListener( "click", function( event )
			{
				self.handleDocumentClick( event );
			} );
			document.addEventListener( "keydown", function( event )
			{
				self.handleKeyboardShortcuts( event );
			} );
			window.addEventListener( "beforeunload", function()
			{
				self.saveNavigationScroll();
			} );
			if( typeof document$ !== "undefined" && document$ &&
					typeof document$.subscribe === "function" )
			{
				document$.subscribe( function()
				{
					document.documentElement.removeAttribute(
							self.config.initializedAttribute );
					setTimeout( function()
					{
						self.init();
					}, 25 );
				} );
			}
		},
		handleDocumentClick: function( event )
		{
			const target = event.target;
			if( !target )
			{
				return;
			}
			if( target.closest && target.closest( "#" + this.config.scrollTopId ) )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( target.closest && target.closest( "[data-loca-copy-heading]" ) )
			{
				event.preventDefault();
				this.copyHeadingLink( target.closest( "[data-loca-copy-heading]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-loca-copy-page]" ) )
			{
				event.preventDefault();
				this.copyPageLink( target.closest( "[data-loca-copy-page]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-loca-print-page]" ) )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( target.closest && target.closest( "[data-loca-toggle-code]" ) )
			{
				event.preventDefault();
				this.toggleCodeBlock( target.closest( "[data-loca-toggle-code]" ) );
				return;
			}
			if( target.closest && target.closest( "[data-loca-api-expand]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( true );
				return;
			}
			if( target.closest && target.closest( "[data-loca-api-collapse]" ) )
			{
				event.preventDefault();
				this.setApiDetailsState( false );
				return;
			}
			if( target.closest && target.closest( "[data-loca-api-clear]" ) )
			{
				event.preventDefault();
				this.clearApiFilter();
				return;
			}
		},
		handleKeyboardShortcuts: function( event )
		{
			const key = String( event.key || "" ).toLowerCase();
			if( event.altKey && key === "t" )
			{
				event.preventDefault();
				this.scrollToTop();
				return;
			}
			if( event.altKey && key === "p" )
			{
				event.preventDefault();
				window.print();
				return;
			}
			if( event.altKey && key === "l" )
			{
				event.preventDefault();
				this.copyCurrentPageToClipboard();
				return;
			}
			if( event.altKey && key === "f" )
			{
				const apiSearch = document.getElementById( this.config.apiSearchId );
				if( apiSearch )
				{
					event.preventDefault();
					apiSearch.focus();
				}
			}
		},
		enhanceExternalLinks: function()
		{
			const links = document.querySelectorAll( ".md-typeset a[href]" );
			const currentHost = window.location.host;
			links.forEach( function( link )
			{
				try
				{
					const url = new URL( link.href, window.location.href );
					if( url.host && url.host !== currentHost )
					{
						link.setAttribute( "target", "_blank" );
						link.setAttribute( "rel", "noopener noreferrer" );
						link.classList.add( LocaDocs.config.externalLinkClass );
						if( !link.querySelector( ".loca-external-indicator" ) )
						{
							const indicator = document.createElement( "span" );
							indicator.className = "loca-external-indicator";
							indicator.setAttribute( "aria-hidden", "true" );
							indicator.textContent = " ↗";
							link.appendChild( indicator );
						}
					}
				}
				catch( error )
				{
					return;
				}
			} );
		},
		customizeSearch: function()
		{
			const searchInputs = document.querySelectorAll( "input.md-search__input" );
			searchInputs.forEach( function( input )
			{
				input.setAttribute( "placeholder", "Search Loca docs..." );
				input.setAttribute( "aria-label", "Search Loca documentation" );
			} );
		},
		addReadingProgress: function()
		{
			if( document.getElementById( this.config.progressId ) )
			{
				return;
			}
			const progress = document.createElement( "div" );
			progress.id = this.config.progressId;
			progress.setAttribute( "aria-hidden", "true" );
			progress.innerHTML = "<span></span>";
			document.body.appendChild( progress );
		},
		updateReadingProgress: function()
		{
			const progress = document.querySelector(
					"#" + this.config.progressId + " span" );
			if( !progress )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			const scrollTop = window.scrollY || document.documentElement.scrollTop;
			if( content )
			{
				const rect = content.getBoundingClientRect();
				const contentTop = rect.top + scrollTop;
				const contentHeight = Math.max( content.offsetHeight, 1 );
				const contentScroll = Math.min( Math.max( scrollTop - contentTop, 0 ),
						contentHeight );
				const percent = Math.min( Math.max( contentScroll / contentHeight, 0 ), 1 );
				progress.style.width = ( percent * 100 ).toFixed( 2 ) + "%";
				return;
			}
			let maxScroll = document.documentElement.scrollHeight - window.innerHeight;
			if( maxScroll <= 0 )
			{
				maxScroll = 1;
			}
			progress.style.width =
					Math.min( Math.max( ( scrollTop / maxScroll ) * 100, 0 ), 100 )
							.toFixed( 2 ) + "%";
		},
		addScrollTopButton: function()
		{
			if( document.getElementById( this.config.scrollTopId ) )
			{
				return;
			}
			const button = document.createElement( "button" );
			button.id = this.config.scrollTopId;
			button.type = "button";
			button.className = "loca-scroll-top";
			button.setAttribute( "aria-label", "Scroll to top" );
			button.setAttribute( "title", "Scroll to top (Alt+T)" );
			button.innerHTML = "↑";
			document.body.appendChild( button );
		},
		updateScrollTopVisibility: function()
		{
			const button = document.getElementById( this.config.scrollTopId );
			if( !button )
			{
				return;
			}
			if( ( window.scrollY || document.documentElement.scrollTop ) >
					this.config.scrollTopVisibleAt )
			{
				button.classList.add( "is-visible" );
			}
			else
			{
				button.classList.remove( "is-visible" );
			}
		},
		scrollToTop: function()
		{
			window.scrollTo( {
				top: 0,
				behavior: "smooth"
			} );
		},
		addPagePathMetadata: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content || content.querySelector( ".loca-page-path" ) )
			{
				return;
			}
			const h1 = content.querySelector( "h1" );
			if( !h1 )
			{
				return;
			}
			const path = window.location.pathname
					.replace( /\/$/, "" )
					.split( "/" )
					.filter( Boolean )
					.slice( -4 )
					.join( " / " );
			if( !path )
			{
				return;
			}
			const meta = document.createElement( "div" );
			meta.className = "loca-page-path";
			meta.textContent = "Docs path: " + path;
			h1.insertAdjacentElement( "afterend", meta );
		},
		addPageTools: function()
		{
			if( document.getElementById( this.config.pageToolsId ) )
			{
				return;
			}
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const title = content.querySelector( "h1" );
			if( !title )
			{
				return;
			}
			const tools = document.createElement( "div" );
			tools.id = this.config.pageToolsId;
			tools.className = "loca-page-tools";
			tools.innerHTML = [
				"<button type=\"button\" data-loca-copy-page title=\"Copy page link\" aria-label=\"Copy page link\">Copy link</button>",
				"<button type=\"button\" data-loca-print-page title=\"Print page\" aria-label=\"Print page\">Print</button>"
			].join( "" );
			const existingPath = content.querySelector( ".loca-page-path" );
			if( existingPath )
			{
				existingPath.insertAdjacentElement( "afterend", tools );
			}
			else
			{
				title.insertAdjacentElement( "afterend", tools );
			}
		},
		copyPageLink: function( button )
		{
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		copyCurrentPageToClipboard: function()
		{
			const button = document.querySelector( "[data-loca-copy-page]" );
			this.copyTextToClipboard( window.location.href, button, "Copied", "Copy link" );
		},
		addHeadingLinks: function()
		{
			const headings = document.querySelectorAll( this.config.headingSelector );
			headings.forEach( function( heading )
			{
				if( heading.querySelector( "." + LocaDocs.config.headingLinkClass ) )
				{
					return;
				}
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = LocaDocs.config.headingLinkClass;
				button.setAttribute( "data-loca-copy-heading", heading.id );
				button.setAttribute( "aria-label",
						"Copy link to " + heading.textContent.trim() );
				button.setAttribute( "title", "Copy section link" );
				button.textContent = "§";
				heading.appendChild( button );
			} );
		},
		copyHeadingLink: function( button )
		{
			const id = button.getAttribute( "data-loca-copy-heading" );
			if( !id )
			{
				return;
			}
			const url = window.location.origin
					+ window.location.pathname
					+ window.location.search
					+ "#"
					+ encodeURIComponent( id );
			this.copyTextToClipboard( url, button, "Copied", "§" );
		},
		copyTextToClipboard: function( text, button, successText, defaultText )
		{
			const updateButton = function()
			{
				if( !button )
				{
					return;
				}
				const previous = button.textContent;
				button.textContent = successText || "Copied";
				setTimeout( function()
				{
					button.textContent = defaultText || previous;
				}, 1400 );
			};
			if( navigator.clipboard && typeof navigator.clipboard.writeText === "function" )
			{
				navigator.clipboard.writeText( text ).then( updateButton ).catch( function()
				{
					LocaDocs.fallbackCopyText( text );
					updateButton();
				} );
				return;
			}
			this.fallbackCopyText( text );
			updateButton();
		},
		fallbackCopyText: function( text )
		{
			const textarea = document.createElement( "textarea" );
			textarea.value = text;
			textarea.setAttribute( "readonly", "readonly" );
			textarea.style.position = "fixed";
			textarea.style.top = "-9999px";
			textarea.style.left = "-9999px";
			document.body.appendChild( textarea );
			textarea.select();
			try
			{
				document.execCommand( "copy" );
			}
			catch( error )
			{
				return;
			}
			finally
			{
				document.body.removeChild( textarea );
			}
		},
		addTableFilters: function()
		{
			const tables = document.querySelectorAll( this.config.tableSelector );
			tables.forEach( function( table, index )
			{
				if( table.getAttribute( "data-loca-filtered" ) === "true" )
				{
					return;
				}
				const tbody = table.querySelector( "tbody" );
				if( !tbody )
				{
					return;
				}
				const rows = Array.prototype.slice.call( tbody.querySelectorAll( "tr" ) );
				if( rows.length < LocaDocs.config.largeTableMinimumRows )
				{
					return;
				}
				table.setAttribute( "data-loca-filtered", "true" );
				const wrapper = document.createElement( "div" );
				wrapper.className = LocaDocs.config.tableToolsClass;
				const input = document.createElement( "input" );
				input.type = "search";
				input.className = LocaDocs.config.tableFilterClass;
				input.placeholder = "Filter table...";
				input.setAttribute( "aria-label", "Filter table " + ( index + 1 ) );
				const count = document.createElement( "span" );
				count.className = LocaDocs.config.tableCountClass;
				count.textContent = rows.length + " rows";
				wrapper.appendChild( input );
				wrapper.appendChild( count );
				table.parentNode.insertBefore( wrapper, table );
				input.addEventListener( "input", function()
				{
					LocaDocs.filterTable( table, input.value, count );
				} );
			} );
		},
		filterTable: function( table, query, countElement )
		{
			const normalizedQuery = String( query || "" ).toLowerCase().trim();
			const rows = Array.prototype.slice.call( table.querySelectorAll( "tbody tr" ) );
			let visible = 0;
			rows.forEach( function( row )
			{
				const text = row.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					row.style.display = "";
					visible += 1;
				}
				else
				{
					row.style.display = "none";
				}
			} );
			if( countElement )
			{
				countElement.textContent = visible + " / " + rows.length + " rows";
			}
		},
		addCodeLabels: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-loca-labeled" ) === "true" )
				{
					return;
				}
				const language = LocaDocs.detectCodeLanguage( code );
				if( !language )
				{
					return;
				}
				pre.setAttribute( "data-loca-labeled", "true" );
				const label = document.createElement( "div" );
				label.className = LocaDocs.config.codeLabelClass;
				label.textContent = language;
				pre.insertAdjacentElement( "beforebegin", label );
			} );
		},
		detectCodeLanguage: function( code )
		{
			const className = code.className || "";
			const match = className.match( /language-([a-zA-Z0-9_+-]+)/ );
			if( match && match[ 1 ] )
			{
				return this.formatLanguageName( match[ 1 ] );
			}
			const text = code.textContent.trim();
			if( /^site_name:|^theme:|^plugins:|^nav:|^extra_css:|^extra_javascript:/m.test(
					text ) )
			{
				return "YAML";
			}
			if( /def\s+\w+\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import/.test( text ) )
			{
				return "Python";
			}
			if( /^mkdocs\s|^python\s|-m\s+|^pip\s|^pytest\s|^git\s/m.test( text ) )
			{
				return "Shell";
			}
			if( /^\{[\s\S]*\}$/.test( text ) )
			{
				return "JSON";
			}
			if( /^#\s|^##\s|```/.test( text ) )
			{
				return "Markdown";
			}
			if( /^\s*<[^>]+>/.test( text ) )
			{
				return "HTML/XML";
			}
			if( /^\s*[\w.-]+\s*\{[\s\S]*\}/.test( text ) )
			{
				return "CSS";
			}
			return "";
		},
		formatLanguageName: function( language )
		{
			const map = {
				py: "Python",
				python: "Python",
				ps1: "PowerShell",
				powershell: "PowerShell",
				bash: "Shell",
				sh: "Shell",
				shell: "Shell",
				yaml: "YAML",
				yml: "YAML",
				json: "JSON",
				md: "Markdown",
				markdown: "Markdown",
				html: "HTML",
				xml: "XML",
				css: "CSS",
				js: "JavaScript",
				javascript: "JavaScript",
				text: "Text",
				txt: "Text"
			};
			const key = String( language || "" ).toLowerCase();
			return map[ key ] || key.toUpperCase();
		},
		addCodeToggles: function()
		{
			const codeBlocks = document.querySelectorAll( this.config.codeSelector );
			codeBlocks.forEach( function( code )
			{
				const pre = code.parentElement;
				if( !pre || pre.getAttribute( "data-loca-toggle-ready" ) === "true" )
				{
					return;
				}
				pre.setAttribute( "data-loca-toggle-ready", "true" );
				if( pre.scrollHeight <= LocaDocs.config.maxCollapsedCodeHeight + 80 )
				{
					return;
				}
				pre.classList.add( LocaDocs.config.codeCollapsedClass );
				pre.style.maxHeight = LocaDocs.config.maxCollapsedCodeHeight + "px";
				const button = document.createElement( "button" );
				button.type = "button";
				button.className = LocaDocs.config.codeToggleClass;
				button.setAttribute( "data-loca-toggle-code", "collapsed" );
				button.textContent = "Show full code";
				pre.insertAdjacentElement( "afterend", button );
			} );
		},
		toggleCodeBlock: function( button )
		{
			const pre = button.previousElementSibling;
			if( !pre || pre.tagName.toLowerCase() !== "pre" )
			{
				return;
			}
			const state = button.getAttribute( "data-loca-toggle-code" );
			if( state === "collapsed" )
			{
				pre.classList.remove( this.config.codeCollapsedClass );
				pre.style.maxHeight = "";
				button.setAttribute( "data-loca-toggle-code", "expanded" );
				button.textContent = "Collapse code";
			}
			else
			{
				pre.classList.add( this.config.codeCollapsedClass );
				pre.style.maxHeight = this.config.maxCollapsedCodeHeight + "px";
				button.setAttribute( "data-loca-toggle-code", "collapsed" );
				button.textContent = "Show full code";
			}
		},
		enhanceApiReference: function()
		{
			const apiContainers = document.querySelectorAll(
					this.config.apiObjectSelector );
			apiContainers.forEach( function( container )
			{
				if( container.getAttribute( "data-loca-api-enhanced" ) === "true" )
				{
					return;
				}
				container.setAttribute( "data-loca-api-enhanced", "true" );
				const heading = container.querySelector( "h2, h3, h4, h5" );
				if( !heading ||
						heading.querySelector( "." + LocaDocs.config.apiBadgeClass ) )
				{
					return;
				}
				const badge = document.createElement( "span" );
				badge.className = LocaDocs.config.apiBadgeClass;
				if( container.className.indexOf( "doc-class" ) !== -1 )
				{
					badge.textContent = "class";
				}
				else if( container.className.indexOf( "doc-method" ) !== -1 )
				{
					badge.textContent = "method";
				}
				else if( container.className.indexOf( "doc-function" ) !== -1 )
				{
					badge.textContent = "function";
				}
				else if( container.className.indexOf( "doc-attribute" ) !== -1 )
				{
					badge.textContent = "attribute";
				}
				else if( container.className.indexOf( "doc-property" ) !== -1 )
				{
					badge.textContent = "property";
				}
				else
				{
					badge.textContent = "api";
				}
				heading.appendChild( badge );
			} );
		},
		addApiTools: function()
		{
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			if( content.querySelector( "." + this.config.apiToolsClass ) )
			{
				return;
			}
			const apiObjects = content.querySelectorAll( this.config.apiObjectSelector );
			const detailsBlocks = content.querySelectorAll( "details" );
			if( apiObjects.length === 0 && detailsBlocks.length === 0 )
			{
				return;
			}
			const firstHeading = content.querySelector( "h1" );
			if( !firstHeading )
			{
				return;
			}
			const panel = document.createElement( "section" );
			panel.className = this.config.apiToolsClass;
			panel.setAttribute( "aria-label", "API tools" );
			panel.innerHTML = [
				"<h2 class=\"loca-api-tools-title\">API Tools</h2>",
				"<label class=\"loca-api-search-label\" for=\"" + this.config.apiSearchId +
				"\">Filter classes, functions, methods, properties, or text</label>",
				"<input id=\"" + this.config.apiSearchId +
				"\" class=\"loca-api-search\" type=\"search\" placeholder=\"Filter API reference...\" autocomplete=\"off\">",
				"<div class=\"loca-api-tool-buttons\">",
				"<button type=\"button\" class=\"loca-api-tool-button\" data-loca-api-expand>Expand all</button>",
				"<button type=\"button\" class=\"loca-api-tool-button\" data-loca-api-collapse>Collapse all</button>",
				"<button type=\"button\" class=\"loca-api-tool-button\" data-loca-api-clear>Clear filter</button>",
				"</div>",
				"<p class=\"loca-api-filter-status\" aria-live=\"polite\"></p>"
			].join( "" );
			const pageTools = content.querySelector( "#" + this.config.pageToolsId );
			if( pageTools )
			{
				pageTools.insertAdjacentElement( "afterend", panel );
			}
			else
			{
				firstHeading.insertAdjacentElement( "afterend", panel );
			}
			const input = panel.querySelector( "#" + this.config.apiSearchId );
			const status = panel.querySelector( ".loca-api-filter-status" );
			if( input )
			{
				input.addEventListener( "input", function()
				{
					LocaDocs.filterApiObjects( input.value, status );
				} );
			}
		},
		filterApiObjects: function( query, statusElement )
		{
			const normalizedQuery = String( query || "" ).trim().toLowerCase();
			const content = document.querySelector( this.config.contentSelector );
			if( !content )
			{
				return;
			}
			const objects = Array.prototype.slice.call(
					content.querySelectorAll( this.config.apiObjectSelector ) );
			if( objects.length === 0 )
			{
				if( statusElement )
				{
					statusElement.textContent = "";
				}
				return;
			}
			let visibleCount = 0;
			objects.forEach( function( object )
			{
				const text = object.textContent.toLowerCase();
				if( !normalizedQuery || text.indexOf( normalizedQuery ) !== -1 )
				{
					object.classList.remove( LocaDocs.config.apiHiddenClass );
					visibleCount += 1;
				}
				else
				{
					object.classList.add( LocaDocs.config.apiHiddenClass );
				}
			} );
			if( statusElement )
			{
				if( !normalizedQuery )
				{
					statusElement.textContent = "";
				}
				else
				{
					statusElement.textContent = visibleCount + " matching API sections";
				}
			}
		},
		setApiDetailsState: function( open )
		{
			const detailsBlocks = document.querySelectorAll( ".md-content__inner details" );
			detailsBlocks.forEach( function( details )
			{
				details.open = open;
			} );
		},
		clearApiFilter: function()
		{
			const input = document.getElementById( this.config.apiSearchId );
			const status = document.querySelector( ".loca-api-filter-status" );
			if( !input )
			{
				return;
			}
			input.value = "";
			this.filterApiObjects( "", status );
			input.focus();
		},
		enhanceTocProgress: function()
		{
			const toc = document.querySelector( this.config.tocSelector );
			if( !toc || toc.getAttribute( "data-loca-toc-enhanced" ) === "true" )
			{
				return;
			}
			toc.setAttribute( "data-loca-toc-enhanced", "true" );
			const marker = document.createElement( "div" );
			marker.className = "loca-toc-marker";
			marker.setAttribute( "aria-hidden", "true" );
			toc.appendChild( marker );
		},
		updateTocProgress: function()
		{
			const headings = Array.prototype.slice.call(
					document.querySelectorAll( this.config.headingSelector ) );
			if( headings.length === 0 )
			{
				return;
			}
			let activeHeading = headings[ 0 ];
			const offset = 120;
			headings.forEach( function( heading )
			{
				const rect = heading.getBoundingClientRect();
				if( rect.top <= offset )
				{
					activeHeading = heading;
				}
			} );
			const tocLinks = document.querySelectorAll(
					this.config.tocSelector + " a[href^='#']" );
			const activeId = activeHeading
			                 ? activeHeading.id
			                 : "";
			tocLinks.forEach( function( link )
			{
				const href = decodeURIComponent(
						( link.getAttribute( "href" ) || "" ).replace( /^#/, "" ) );
				if( href === activeId )
				{
					link.classList.add( LocaDocs.config.tocActiveClass );
				}
				else
				{
					link.classList.remove( LocaDocs.config.tocActiveClass );
				}
			} );
		},
		saveNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				window.sessionStorage.setItem( this.config.navScrollKey,
						String( nav.scrollTop || 0 ) );
			}
			catch( error )
			{
				return;
			}
		},
		restoreNavigationScroll: function()
		{
			const nav = document.querySelector( this.config.navSelector );
			if( !nav )
			{
				return;
			}
			try
			{
				const value = window.sessionStorage.getItem( this.config.navScrollKey );
				if( value !== null )
				{
					nav.scrollTop = parseInt( value, 10 ) || 0;
				}
			}
			catch( error )
			{
				return;
			}
		},
		enhanceKeyboardFocus: function()
		{
			document.body.addEventListener( "keydown", function( event )
			{
				if( event.key === "Tab" )
				{
					document.body.classList.add( LocaDocs.config.keyboardModeClass );
				}
			} );
			document.body.addEventListener( "mousedown", function()
			{
				document.body.classList.remove( LocaDocs.config.keyboardModeClass );
			} );
		},
		addMermaidGuard: function()
		{
			const blocks = document.querySelectorAll( "code.language-mermaid" );
			blocks.forEach( function( block )
			{
				block.setAttribute( "data-loca-mermaid-detected", "true" );
			} );
		}
	};
	
	function ready( callback )
	{
		if( document.readyState === "loading" )
		{
			document.addEventListener( "DOMContentLoaded", callback );
		}
		else
		{
			callback();
		}
	}
	
	ready( function()
	{
		LocaDocs.init();
	} );
} )();

