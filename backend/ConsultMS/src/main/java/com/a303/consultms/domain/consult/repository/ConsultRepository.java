package com.a303.consultms.domain.consult.repository;

import com.a303.consultms.domain.consult.Consult;
import com.a303.consultms.domain.consult.dto.response.ConsultDetailRes;
import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ConsultRepository extends JpaRepository<Consult, Integer>,
    ConsultCustomRepository {

    ConsultDetailRes findConsultByConsultId(int consultId);

    //날짜 내림차순으로 정렬
    List<Consult> findAllByOrderByCreateTimeDesc();

    //공유된 고민 리스트 조회, 날짜 내림차순 정렬
    List<Consult> findAllByIsSharedOrderByCreateTimeDesc(boolean isShared);

    //채널ID로 조회
    List<Consult> findAllByChannelIdOrderByCreateTimeDesc(String channelId);

    //대화중인 채팅방 목록 (내가 만든 고민상담소)
    List<Consult> findAllByMemberIdOrderByUpdateTimeDesc(int memberId);

    //대화중인 채팅방 목록 (내가 들어주는 고민상담소)
    List<Consult> findAllByMemberIdNotOrderByUpdateTimeDesc(int memberId);

    List<Consult> findAllByCategoryIdOrderByUpdateTimeDesc(int categoryId);

    List<Consult> findAllByCategoryIdAndIsSharedOrderByUpdateTimeDesc(int categoryId, boolean isShared);

    Consult findByChannelId(String channelId);

}
