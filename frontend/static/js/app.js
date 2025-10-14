document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = window.location.origin;

    const lastSavedResponses = new Map();

    const debounce = (fn, delay = 300) => {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn.apply(null, args), delay);
        };
    };

    const markResponseAsSaved = (id, text) => {
        if (!id) return;
        const trimmed = (text || '').trim();
        if (!trimmed) return;
        lastSavedResponses.set(id, trimmed);
    };

    const saveResponseToCache = async (id, text) => {
        if (!id) return;
        const trimmed = (text || '').trim();
        if (!trimmed) return;
        if (lastSavedResponses.get(id) === trimmed) return;
        lastSavedResponses.set(id, trimmed);
        try {
            await fetch(`${API_BASE_URL}/api/cache-selected-response`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, response: trimmed })
            });
        } catch (error) {
            console.error('Не удалось сохранить ответ в кеш:', error);
        }
    };

    // --- State Management ---
    const state = {
        feedbacks: {
            items: [],
            currentIndex: 0,
            container: document.getElementById('feedbacks-slider'),
            counter: document.getElementById('feedbacks-counter'),
            loader: document.getElementById('feedbacks-loader'),
            emptyState: document.getElementById('feedbacks-empty'),
            nav: document.getElementById('feedbacks-nav')
        },
        questions: {
            items: [],
            currentIndex: 0,
            container: document.getElementById('questions-slider'),
            counter: document.getElementById('questions-counter'),
            loader: document.getElementById('questions-loader'),
            emptyState: document.getElementById('questions-empty'),
            nav: document.getElementById('questions-nav')
        }
    };

    const updateNavVisibility = (type) => {
        const typeState = state[type];
        if (!typeState || !typeState.nav) return;
        const tabElement = document.getElementById(type);
        const isActiveTab = tabElement ? tabElement.classList.contains('active') : false;
        const hasItems = Array.isArray(typeState.items) && typeState.items.length > 0;
        typeState.nav.style.display = isActiveTab && hasItems ? 'flex' : 'none';
    };

    // --- Main App Initialization ---
    const init = () => {
        setupTabs();
        setupNavButtons();
        setupAnimations();
        setupImageOverlay();
        fetchDataForBothTabs();
        window.addEventListener('resize', () => {
            renderAllItems('feedbacks');
            renderAllItems('questions');
        });
    };

    const setupAnimations = () => {
        document.body.classList.add('loaded');
        document.querySelector('.app-container').classList.add('appear');
    };

    const setupImageOverlay = () => {
        const overlay = document.getElementById('image-overlay');
        const closeBtn = overlay.querySelector('.close-btn');

        const closeOverlay = () => overlay.classList.remove('visible');

        closeBtn.addEventListener('click', closeOverlay);
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeOverlay();
            }
        });
    };

    // --- Setup Functions ---
    const setupTabs = () => {
        const STORAGE_ACTIVE_TAB = 'wb_active_tab';
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tab = button.dataset.tab;
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                document.getElementById(tab).classList.add('active');
                tabButtons.forEach(b => b.classList.remove('active'));
                button.classList.add('active');

                updateNavVisibility('feedbacks');
                updateNavVisibility('questions');

                requestAnimationFrame(() => {
                    const typeState = state[tab];
                    if (typeState && Array.isArray(typeState.items) && typeState.items.length > 0) {
                        renderAllItems(tab);
                    }
                });

                try { localStorage.setItem(STORAGE_ACTIVE_TAB, tab); } catch (_) {}
            });
        });

        // Restore active tab from storage
        try {
            const savedTab = localStorage.getItem(STORAGE_ACTIVE_TAB);
            if (savedTab && savedTab !== 'feedbacks') {
                const button = Array.from(tabButtons).find(b => b.dataset.tab === savedTab);
                if (button) {
                    button.click();
                }
            }
        } catch (_) {}

        updateNavVisibility('feedbacks');
        updateNavVisibility('questions');
    };

    const setupNavButtons = () => {
        document.querySelectorAll('.nav-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const type = button.dataset.type;
                const isNext = button.classList.contains('next');
                if (isNext) {
                    navigateTo(type, state[type].currentIndex + 1);
                } else {
                    navigateTo(type, state[type].currentIndex - 1);
                }
            });
        });
    };

    // --- Data Fetching and Rendering ---
    const fetchDataForBothTabs = async () => {
        await fetchAndDisplayItems('feedbacks');
        await fetchAndDisplayItems('questions');
    };

    const fetchAndDisplayItems = async (type) => {
        const getSavedIdKey = (t) => `wb_current_id_${t}`;
        const typeState = state[type];
        typeState.loader.style.display = 'flex';
        typeState.nav.style.display = 'none';
        typeState.container.innerHTML = '';

        try {
            const response = await fetch(`${API_BASE_URL}/api/${type}?v=${new Date().getTime()}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const items = await response.json();
            
            typeState.items = items;
            if (items.length > 0) {
                // Restore previously selected item if still exists
                let index = 0;
                try {
                    const savedId = localStorage.getItem(getSavedIdKey(type));
                    if (savedId) {
                        const found = items.findIndex(i => i.id === savedId);
                        if (found >= 0) index = found;
                    }
                } catch (_) {}
                typeState.currentIndex = index;
                renderAllItems(type);
                updateNavVisibility(type);
                typeState.emptyState.style.display = 'none';
            } else {
                typeState.emptyState.style.display = 'block';
                updateNavVisibility(type);
                updateNavButtons(type);
                updateCounter(type);
                try { localStorage.removeItem(getSavedIdKey(type)); } catch (_) {}
            }
        } catch (error) {
            console.error(`Could not fetch ${type}:`, error);
            typeState.emptyState.innerHTML = `<p>Ошибка загрузки данных.</p>`;
            typeState.items = [];
            typeState.emptyState.style.display = 'block';
            updateNavVisibility(type);
        } finally {
            typeState.loader.style.display = 'none';
            updateNavVisibility(type);
        }
    };
    
    const renderAllItems = (type) => {
        const typeState = state[type];
        const slider = typeState.container;
        const viewport = slider.parentElement;
        
        slider.innerHTML = '';

        if (!viewport) return;

        const cardWidth = viewport.getBoundingClientRect().width;

        typeState.items.forEach(item => {
            const card = createItemCard(item, type);
            card.style.width = `${cardWidth}px`;
            slider.appendChild(card);
        });

        if (typeState.items.length > 0) {
            navigateTo(type, typeState.currentIndex);
        }

        updateNavVisibility(type);
    };

    const createItemCard = (item, type) => {
        const template = document.getElementById('item-template');
        const cardTemplate = template.content.cloneNode(true);
        const cardElement = cardTemplate.querySelector('.item-card');

        populateCard(cardTemplate, item, type);
        
        const regenerateBtn = cardTemplate.querySelector('.regenerate-btn');
        const sendBtn = cardTemplate.querySelector('.send-btn');
        const promptInput = cardTemplate.querySelector('.prompt-input');
        const responseTextarea = cardTemplate.querySelector('.response-textarea');

        // Авто-рост textarea по содержимому
        const autoResize = () => {
            responseTextarea.style.height = 'auto';
            responseTextarea.style.height = `${responseTextarea.scrollHeight}px`;
        };
        const manualSaveDebounced = debounce(() => saveResponseToCache(item.id, responseTextarea.value), 800);
        responseTextarea.addEventListener('input', () => {
            autoResize();
            manualSaveDebounced();
        });
        responseTextarea.addEventListener('blur', () => {
            saveResponseToCache(item.id, responseTextarea.value);
        });
        // Первичная подгонка
        requestAnimationFrame(autoResize);

        promptInput.addEventListener('input', () => {
            const buttonText = promptInput.value.trim() ? 'Редактировать' : 'Повторить';
            regenerateBtn.innerHTML = `<i class="fa-solid fa-sync"></i> ${buttonText}`;
        });

        regenerateBtn.addEventListener('click', () => {
            const textForAI = type === 'feedbacks' ? getFullFeedbackText(item) : item.text;
            const rating = type === 'feedbacks' ? item.productValuation : null;
            const customPrompt = promptInput.value.trim();
            const productName = item.productDetails.productName;
            const pluses = type === 'feedbacks' ? (item.pluses || null) : null;
            const minuses = type === 'feedbacks' ? (item.minuses || null) : null;

            if (promptInput.value.trim()) {
                // Изменить: мгновенно заменить без печатания
                generateResponse(item.id, textForAI, responseTextarea, promptInput.value, rating, true, productName, false, item.advantages || null, pluses, minuses);
        } else {
                generateMultipleResponses(item.id, textForAI, responseTextarea, null, rating, productName, item.advantages || null, pluses, minuses);
            }
        });
        
        sendBtn.addEventListener('click', () => {
            const responseText = responseTextarea.value.trim();
            if (!responseText) {
                showToast("Нельзя отправить пустой ответ.", "error");
                return;
            }
            sendBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Отправка...';
            sendBtn.disabled = true;
            
            handleSend(item.id, type, responseText, sendBtn, cardElement);
        });

        const textForAI = type === 'feedbacks' ? getFullFeedbackText(item) : item.text;
        const rating = type === 'feedbacks' ? item.productValuation : null;
        const productName = item.productDetails.productName;
        const pluses = type === 'feedbacks' ? (item.pluses || null) : null;
        const minuses = type === 'feedbacks' ? (item.minuses || null) : null;
        
        // Первая генерация: без печатной анимации (чтобы не тормозить при перезагрузке)
        generateResponse(item.id, textForAI, responseTextarea, null, rating, false, productName, false, item.advantages || null, pluses, minuses);

        return cardElement;
    };

    const getFullFeedbackText = (item) => {
        let htmlContent = '';
        
        // Добавляем комментарий, если есть основной текст
        if (item.text && item.text.trim()) {
            htmlContent += `<div class="feedback-comment"><span class="feedback-label comment-label">Комментарий:</span> ${item.text.trim()}</div>`;
        }
        
        // Добавляем достоинства
        if (item.pluses && item.pluses.trim()) {
            htmlContent += `<div class="feedback-pluses"><span class="feedback-label pluses-label">Достоинства:</span> ${item.pluses.trim()}</div>`;
        }
        
        // Добавляем недостатки
        if (item.minuses && item.minuses.trim()) {
            htmlContent += `<div class="feedback-minuses"><span class="feedback-label minuses-label">Недостатки:</span> ${item.minuses.trim()}</div>`;
        }
        
        return htmlContent || '<div class="feedback-empty">Отзыв без текста.</div>';
    };

    // Функция для подсчета компонентов отзыва
    const countFeedbackComponents = (item) => {
        let count = 0;
        if (item.text && item.text.trim()) count++; // Комментарий
        if (item.pluses && item.pluses.trim()) count++; // Достоинства
        if (item.minuses && item.minuses.trim()) count++; // Недостатки
        return count;
    };

    // Функция для обновления отступа навигации
    const updateNavigationBottom = (type) => {
        const currentItem = state[type].items[state[type].currentIndex];
        if (!currentItem) return;
        
        const componentCount = type === 'feedbacks' ? countFeedbackComponents(currentItem) : 1;
        let bottomValue;
        
        switch (componentCount) {
            case 1:
                bottomValue = '50px';
                break;
            case 2:
                bottomValue = '20px';
                break;
            case 3:
                bottomValue = '-10px';
                break;
            default:
                bottomValue = '50px';
        }
        
        const nav = state[type].nav;
        if (nav) {
            nav.style.bottom = bottomValue;
        }
    };

    const populateCard = (card, item, type) => {
        card.querySelector('.product-link').href = `https://www.wildberries.ru/catalog/${item.productDetails.nmId}/detail.aspx`;
        card.querySelector('.name-text').textContent = item.productDetails.productName;
        card.querySelector('.date-text').textContent = new Date(item.createdDate).toLocaleDateString('ru-RU');
        card.querySelector('.item-text').innerHTML = type === 'feedbacks' ? getFullFeedbackText(item) : item.text;

        const thumbnail = card.querySelector('.product-thumbnail');
        if (item.productDetails.photo) {
            thumbnail.src = item.productDetails.photo;
            thumbnail.addEventListener('click', () => {
                const overlay = document.getElementById('image-overlay');
                const overlayImg = document.getElementById('overlay-img');
                overlayImg.src = item.productDetails.photo.replace('/tm/', '/big/');
                overlay.classList.add('visible');
            });
        } else {
            thumbnail.style.display = 'none';
        }

        const ratingSpan = card.querySelector('.item-rating');
        if (type === 'feedbacks' && item.productValuation) {
            const rating = parseInt(item.productValuation);
            let stars = Array(5).fill(0).map((_, i) => 
                `<i class="fa-${i < rating ? 'solid' : 'regular'} fa-star" style="color: ${i < rating ? 'gold' : 'var(--secondary-text)'};"></i>`
            ).join('');
            ratingSpan.innerHTML = `<span class="stars-container">${stars}</span> <span class="rating-value">${rating}/5</span>`;
        } else {
            ratingSpan.style.display = 'none';
        }
        
        // Преимущества (чипсы)
        const advSection = card.querySelector('.advantages-section');
        const advRoot = card.querySelector('.advantages-chips');
        if (type === 'feedbacks' && Array.isArray(item.advantages) && item.advantages.length > 0) {
            advSection.style.display = 'block';
            advRoot.innerHTML = '';
            item.advantages.forEach((a) => {
                const chip = document.createElement('span');
                chip.className = 'adv-chip';
                chip.textContent = a;
                advRoot.appendChild(chip);
            });
        } else {
            advSection.style.display = 'none';
        }

        // Обновляем отступ навигации после заполнения карточки
        setTimeout(() => updateNavigationBottom(type), 0);
    };

    const handleSend = async (itemId, type, text, sendBtn, cardElement) => {
        try {
            const payload = { id: itemId, type };
            if (type === 'questions') {
                payload.answer = { text };
                payload.state = 'wbRu';
            } else {
                payload.text = text;
            }

            const response = await fetch(`${API_BASE_URL}/api/reply`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                let errorMsg = 'Не удалось отправить ответ.';
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.detail || errorMsg;
                } catch (_) {}
                throw new Error(errorMsg);
            }

            sendBtn.innerHTML = '<i class="fa-solid fa-check"></i> Отправлено';
            cardElement.classList.add('sent');

            setTimeout(() => {
                const typeState = state[type];
                const index = typeState.items.findIndex(i => i.id === itemId);
                if (index < 0) return;

                const productName = typeState.items[index].productDetails.productName;
                typeState.items.splice(index, 1);

                renderAllItems(type);
                
                if (typeState.items.length === 0) {
                    typeState.emptyState.style.display = 'block';
                    updateNavVisibility(type);
                    updateCounter(type);
                } else {
                    let newIndex = index;
                    if (newIndex >= typeState.items.length) {
                        newIndex = typeState.items.length - 1;
                    }
                    navigateTo(type, newIndex);
                }

                showToast(`Ответ на отзыв о "${productName}" успешно отправлен.`);
            }, 500);

        } catch (error) {
            showToast(error.message, 'error');
            sendBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Ответить';
            sendBtn.disabled = false;
        }
    };

    const navigateTo = (type, index) => {
        const getSavedIdKey = (t) => `wb_current_id_${t}`;
        const typeState = state[type];
        const itemsCount = typeState.items.length;

        if (itemsCount === 0) {
             return;
        }
        
        let newIndex = index;
        if (index < 0) {
            newIndex = itemsCount - 1;
        } else if (index >= itemsCount) {
            newIndex = 0;
        }

        typeState.currentIndex = newIndex;
        
        const slider = typeState.container;
        // For precise alignment, measure the card width from the first card
        const firstCard = slider.querySelector('.item-card');
        let cardWidth = 0;
        if (firstCard) {
            cardWidth = firstCard.getBoundingClientRect().width;
        } else if (slider.parentElement) {
            cardWidth = slider.parentElement.getBoundingClientRect().width;
        }
        const offset = -newIndex * cardWidth;
        slider.style.transform = `translateX(${offset}px)`;

        updateCounter(type);
        updateNavButtons(type);
        updateNavigationBottom(type);
        updateNavVisibility(type)

        // Persist the currently selected item id
        try {
            const currentItem = typeState.items[newIndex];
            if (currentItem && currentItem.id) {
                localStorage.setItem(getSavedIdKey(type), currentItem.id);
            }
        } catch (_) {}
    };

    const updateCounter = (type) => {
        const typeState = state[type];
        const total = typeState.items.length;
        const current = total > 0 ? typeState.currentIndex + 1 : 0;
        typeState.counter.textContent = `${current} / ${total}`;
    };

    const updateNavButtons = (type) => {
        const { currentIndex, items, nav } = state[type];
        const prevBtn = nav.querySelector('.prev');
        const nextBtn = nav.querySelector('.next');
        
        if (!prevBtn || !nextBtn) return;

        // Infinite loop means buttons are only disabled if there's <= 1 item.
        const disabled = items.length <= 1;
        prevBtn.disabled = disabled;
        nextBtn.disabled = disabled;
    };

    // Функция анимации печатной машинки
    const typewriterEffect = (textarea, text, speed = 30) => {
        return new Promise((resolve) => {
            textarea.value = '';
            textarea.classList.add('typewriter-active');
            let i = 0;
            
            const typeInterval = setInterval(() => {
                if (i < text.length) {
                    textarea.value += text.charAt(i);
                    i++;
                    // Автоматическая прокрутка к концу текста
                    textarea.scrollTop = textarea.scrollHeight;
                } else {
                    clearInterval(typeInterval);
                    textarea.classList.remove('typewriter-active');
                    textarea.classList.add('typewriter-complete');
                    
                    // Убираем класс завершения через 1 секунду
                    setTimeout(() => {
                        textarea.classList.remove('typewriter-complete');
                    }, 1000);
                    
                    // Финальная прокрутка к концу
                    setTimeout(() => {
                        textarea.scrollTop = textarea.scrollHeight;
                    }, 100);
                    
                    resolve();
                }
            }, speed);
        });
    };

    const generateMultipleResponses = async (id, text, textarea, prompt, rating, productName, advantages = null, pluses = null, minuses = null) => {
        // Открываем модалку выбора
        const modal = document.getElementById('ai-modal');
        const choicesRoot = document.getElementById('ai-choices');
        const closeBtn = document.getElementById('ai-modal-close');
        if (!modal || !choicesRoot) return;
        document.body.classList.add('modal-open');
        modal.classList.add('visible');
        // Рисуем скелетоны на время загрузки
        choicesRoot.classList.add('ai-skeleton');
        choicesRoot.innerHTML = '';
        for (let i = 0; i < 3; i++) {
            const sk = document.createElement('div');
            sk.className = 'ai-skeleton-card';
            sk.innerHTML = `
                <div class="skeleton skeleton-title"></div>
                <div class="skeleton skeleton-paragraph">
                    <div class="skeleton skeleton-line"></div>
                    <div class="skeleton skeleton-line" style="width: 95%"></div>
                    <div class="skeleton skeleton-line" style="width: 90%"></div>
                    <div class="skeleton skeleton-line" style="width: 85%"></div>
                    <div class="skeleton skeleton-line" style="width: 80%"></div>
                </div>
                <div class="skeleton skeleton-button"></div>
            `;
            choicesRoot.appendChild(sk);
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/generate-multiple-responses`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    id, 
                    text, 
                    prompt, 
                    rating, 
                    productName,
                    advantages,
                    pluses,
                    minuses,
                    force: true  // Всегда принудительно генерируем новые варианты для кнопки "Повторить"
                })
            });

            if (!response.ok) throw new Error('Ошибка сети при запросе нескольких вариантов');
            const data = await response.json();

            choicesRoot.classList.remove('ai-skeleton');
            choicesRoot.innerHTML = '';
            Object.entries(data).forEach(([modelName, resp], index) => {
                const card = document.createElement('div');
                card.className = 'ai-choice-card';
                card.innerHTML = `
                    <div class="ai-choice-header"><i class="fa-solid fa-wand-magic-sparkles"></i> Вариант ${index + 1}</div>
                    <div class="ai-choice-body">${resp}</div>
                    <div class="ai-choice-actions">
                        <button class="ai-choice-select"><i class="fa-solid fa-check"></i> Выбрать</button>
                    </div>
                `;
                const selectBtn = card.querySelector('.ai-choice-select');
                selectBtn.addEventListener('click', () => {
                    // Выбор варианта: мгновенная вставка без печатания
                    textarea.value = resp;
                    markResponseAsSaved(id, resp);
                    fetch(`${API_BASE_URL}/api/cache-selected-response`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id: id, response: resp })
                    }).catch(() => {});
                    modal.classList.remove('visible');
                    document.body.classList.remove('modal-open');
                });
                choicesRoot.appendChild(card);
            });

        } catch (error) {
            console.error('Ошибка при генерации ответов:', error);
            choicesRoot.classList.remove('ai-skeleton');
            choicesRoot.innerHTML = '<div class="error-message">Ошибка при генерации ответов</div>';
        }

        // Закрытие модалки
        const onClose = () => {
            modal.classList.remove('visible');
            document.body.classList.remove('modal-open');
            closeBtn?.removeEventListener('click', onClose);
            modal?.removeEventListener('click', onBackdropClick);
        };
        let lastBackdropClickTs = 0;
        const onBackdropClick = (e) => {
            // Закрываем только если кликнули по фону (не по контенту)
            if (e.target !== modal) return;
            const now = Date.now();
            if (now - lastBackdropClickTs < 2000) {
                onClose();
                lastBackdropClickTs = 0;
            } else {
                lastBackdropClickTs = now;
            }
        };
        closeBtn?.addEventListener('click', onClose);
        modal?.addEventListener('click', onBackdropClick);
    };

    const generateResponse = async (id, text, textarea, prompt = null, rating = null, force = false, productName = '', useTypewriter = true, advantages = null, pluses = null, minuses = null) => {
        textarea.classList.add('generating');
        textarea.placeholder = "ИИ думает...";
        textarea.value = ''; // Очищаем поле перед генерацией

        try {
            const response = await fetch(`${API_BASE_URL}/api/generate-response`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id, text, prompt, rating, force, productName, advantages, pluses, minuses })
            });
            if (!response.ok) throw new Error('Ошибка сети');
            const data = await response.json();
            
            // Убираем индикатор загрузки
            textarea.classList.remove('generating');
            textarea.placeholder = "Здесь будет сгенерированный ответ...";
            
            if (useTypewriter) {
                await typewriterEffect(textarea, data.response);
            } else {
                textarea.value = data.response;
            }
            markResponseAsSaved(id, data.response);
            // Подгоняем высоту после вставки текста
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight}px`;
            
        } catch (error) {
            console.error('Error generating response:', error);
            textarea.classList.remove('generating');
            textarea.placeholder = "Здесь будет сгенерированный ответ...";
            
            // Анимация для сообщения об ошибке
            await typewriterEffect(textarea, "Не удалось сгенерировать ответ. Попробуйте снова.", 50);
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight}px`;
        }
    };

    const showToast = (message, type = 'success') => {
        const toast = document.createElement('div');
        toast.className = `toast-notification ${type}`;
        toast.innerHTML = `<i class="fa-solid ${type === 'success' ? 'fa-check-circle' : 'fa-times-circle'}"></i> ${message}`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('visible');
        }, 10);

        setTimeout(() => {
            toast.classList.remove('visible');
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 500);
            }, 3000);
    };

    // --- Kick it off ---
    init();
});