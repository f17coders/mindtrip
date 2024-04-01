package com.a303.consultms.global.config.websocket;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.ChannelRegistration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.EnableWebSocketMessageBroker;
import org.springframework.web.socket.config.annotation.StompEndpointRegistry;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.config.annotation.WebSocketMessageBrokerConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketTransportRegistration;

@Configuration
@RequiredArgsConstructor
@EnableWebSocketMessageBroker //WebSocket 활성화하고 메시지 브로커 사용가능
//@EnableWebSocket
public class WebSocketConfiguration implements WebSocketMessageBrokerConfigurer  {
//    private final StompHandler stompHandler;

    // 메시지 브로커를 구성하는 메서드
    @Override
    public void configureMessageBroker(MessageBrokerRegistry registry) {
        //메시지를 구독 요청 url -> 메시지 받을 때
        registry.enableSimpleBroker("/sub");
        //메시지를 발행하는 요청 url -> 메시지 보낼 때
        registry.setApplicationDestinationPrefixes("/pub");

        registry.setPreservePublishOrder(true);
    }

    // STOMP 엔드포인트를 등록하는 메서드
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/api/chat") //STOMP 엔드포인트 설정
//            .setAllowedOrigins("*");
            .setAllowedOriginPatterns("http://localhost:5173", "https://mindtrip.site")
            .withSockJS(); //SockJS 사용가능 설정
    }

    // 클라이언트 인바운드 채널을 구성하는 메서드
    @Override
    public void configureClientInboundChannel(ChannelRegistration registration) {
        // stompHandler를 인터셉터로 등록하여 STOMP 메시지 핸들링을 수행
//        registration.interceptors(stompHandler);
    }

    // STOMP에서 64KB 이상의 데이터 전송을 못하는 문제 해결
    @Override
    public void configureWebSocketTransport(WebSocketTransportRegistration registry) {
        registry.setMessageSizeLimit(160 * 64 * 1024);
        registry.setSendTimeLimit(100 * 10000);
        registry.setSendBufferSizeLimit(3 * 512 * 1024);
    }

}
